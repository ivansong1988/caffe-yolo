#ifdef USE_OPENCV
#include <opencv2/core/core.hpp>
#endif  // USE_OPENCV
#include <stdint.h>

#include <vector>

#include "caffe/data_transformer.hpp"
#include "caffe/layers/box_data_layer.hpp"
#include "caffe/util/benchmark.hpp"

namespace caffe {

template <typename Dtype>
BoxDataLayer<Dtype>::BoxDataLayer(const LayerParameter& param)
  : BasePrefetchingDataLayer<Dtype>(param),
    reader_(param) {
}

template <typename Dtype>
BoxDataLayer<Dtype>::~BoxDataLayer() {
  this->StopInternalThread();
}

template <typename Dtype>
void BoxDataLayer<Dtype>::DataLayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  this->box_label_ = true;
  const DataParameter param = this->layer_param_.data_param();
  const int batch_size = param.batch_size();
  // Read a data point, and use it to initialize the top blob.
  Datum& datum = *(reader_.full().peek());

  // Use data_transformer to infer the expected blob shape from datum.
  vector<int> top_shape = this->data_transformer_->InferBlobShape(datum);
  this->transformed_data_.Reshape(top_shape);
  // Reshape top[0] and prefetch_data according to the batch_size.
  top_shape[0] = batch_size;
  top[0]->Reshape(top_shape);
  for (int i = 0; i < this->PREFETCH_COUNT; ++i) {
    this->prefetch_[i].data_.Reshape(top_shape);
  }
  LOG(INFO) << "output data size: " << top[0]->num() << ","
      << top[0]->channels() << "," << top[0]->height() << ","
      << top[0]->width();
  // label
  if (this->output_labels_) {
    if (param.side_size() > 0) {
      for (int i = 0; i < param.side_size(); ++i) {
        sides_.push_back(param.side(i));
      }
    }
    if (sides_.size() == 0) {
      sides_.push_back(7);
    }
    CHECK_EQ(sides_.size(), top.size() - 1) << 
      "side num not equal to top size";
    for (int i = 0; i < this->PREFETCH_COUNT; ++i) {
      this->prefetch_[i].multi_label_.clear(); //prefetch_是队列数据的实际存放空间, Forward时直接将数据从队列中copy到top, 因此在这里与top一并做Reshape
    }
    for (int i = 0; i < sides_.size(); ++i) {
      vector<int> label_shape(1, batch_size);
      int label_size = sides_[i] * sides_[i] * (1 + 1 + 1 + 4);//这里的1+1+1为每个location的"difficult", 'isobj', 'class_label'
      label_shape.push_back(label_size);
      top[i+1]->Reshape(label_shape);
      for (int j = 0; j < this->PREFETCH_COUNT; ++j) {
        shared_ptr<Blob<Dtype> > tmp_blob;
        tmp_blob.reset(new Blob<Dtype>(label_shape));
        this->prefetch_[j].multi_label_.push_back(tmp_blob);
      }
    }
  }
}

// This function is called on prefetch thread
template<typename Dtype>
void BoxDataLayer<Dtype>::load_batch(Batch<Dtype>* batch) {//给数据读取线程调用，往队列prefetch_full_中填入数据
  CPUTimer batch_timer;
  batch_timer.Start();
  double read_time = 0;
  double trans_time = 0;
  CPUTimer timer;
  CHECK(batch->data_.count());
  CHECK(this->transformed_data_.count());

  // Reshape according to the first datum of each batch
  // on single input batches allows for inputs of varying dimension.
  const int batch_size = this->layer_param_.data_param().batch_size();
  Datum& datum = *(reader_.full().peek());
  // Use data_transformer to infer the expected blob shape from datum.
  vector<int> top_shape = this->data_transformer_->InferBlobShape(datum);
  this->transformed_data_.Reshape(top_shape);
  // Reshape batch according to the batch_size.
  top_shape[0] = batch_size;
  batch->data_.Reshape(top_shape);

  Dtype* top_data = batch->data_.mutable_cpu_data();//同理, top_data指向batch中分配好的对应内存
  vector<Dtype*> top_label;

  //预先将batch中的multi_label_数据指针存入top_label，后面将直接利用top_label对batch的对应内存区域进行操作
  if (this->output_labels_) {
    for (int i = 0; i < sides_.size(); ++i) {
      top_label.push_back(batch->multi_label_[i]->mutable_cpu_data());
    }
  }
  for (int item_id = 0; item_id < batch_size; ++item_id) {
    timer.Start();
    // get a datum
    Datum& datum = *(reader_.full().pop("Waiting for data"));//依次从reader_中取出样本(data+标注信息)
    read_time += timer.MicroSeconds();
    timer.Start();
    // Apply data transformations (mirror, scale, crop...)
    int offset = batch->data_.offset(item_id);//默认是第一个参数指定num, 因此每次移动1个样本
    vector<BoxLabel> box_labels;

    //此时top_data相当于batch的一个指示器, 现在将this->transformed_data_的cpu_ptr_指向top_data(batch)内存, 应该是基于共享内存的考虑(不用临时计算出来再拷贝)    
    this->transformed_data_.set_cpu_data(top_data + offset);

    if (this->output_labels_) {
      // rand sample a patch, adjust box labels
      //Transform这个函数是专为目标检测(yolo)的label数据重载的, 具体目标检测的caffe代码该怎么改data_layer也可以同时参考faster-rcnn/ssd
      this->data_transformer_->Transform(datum, &(this->transformed_data_), &box_labels);
      // transform label
      for (int i = 0; i < sides_.size(); ++i) {
        int label_offset = batch->multi_label_[i]->offset(item_id);//默认是第一个参数指定num, 因此每次移动1个样本
        int count  = batch->multi_label_[i]->count(1);//取出元素个数, multi_label_[i]的shape已经在DataLayerSetUp中设置过了
        transform_label(count, top_label[i] + label_offset, box_labels, sides_[i]);//caffe-yolo
      }
    } else {
      this->data_transformer_->Transform(datum, &(this->transformed_data_));
    }
    trans_time += timer.MicroSeconds();

    reader_.free().push(const_cast<Datum*>(&datum));
  }
  timer.Stop();
  batch_timer.Stop();
  DLOG(INFO) << "Prefetch batch: " << batch_timer.MilliSeconds() << " ms.";
  DLOG(INFO) << "     Read time: " << read_time / 1000 << " ms.";
  DLOG(INFO) << "Transform time: " << trans_time / 1000 << " ms.";
}

template<typename Dtype>
void BoxDataLayer<Dtype>::transform_label(int count, Dtype* top_label,
    const vector<BoxLabel>& box_labels, int side) {
  int locations = pow(side, 2);
  CHECK_EQ(count, locations * 7) <<
    "side and count not match";
  // difficult
  caffe_set(locations, Dtype(0), top_label);//每个location存储1+1+1+4, 依次排列
  // isobj
  caffe_set(locations, Dtype(0), top_label + locations);
  // class label
  caffe_set(locations, Dtype(-1), top_label + locations * 2);
  // box
  caffe_set(locations*4, Dtype(0), top_label + locations * 3);
  for (int i = 0; i < box_labels.size(); ++i) {
    float difficult = box_labels[i].difficult_;
    if (difficult != 0. && difficult != 1.) {
      LOG(WARNING) << "Difficult must be 0 or 1";
    }
    float class_label = box_labels[i].class_label_;
    CHECK_GE(class_label, 0) << "class_label must >= 0";
    float x = box_labels[i].box_[0];
    float y = box_labels[i].box_[1];
    // LOG(INFO) << "x: " << x << " y: " << y;
    int x_index = floor(x * side);//判定中心点落入哪个Cell, 其实这里的实现就是要和算法的保持一致就OK
    int y_index = floor(y * side);
    x_index = std::min(x_index, side - 1);//这里强制要求不能超过side - 1
    y_index = std::min(y_index, side - 1);
    int dif_index = side * y_index + x_index;
    int obj_index = locations + dif_index;
    int class_index = locations * 2 + dif_index;
    int cor_index = locations * 3 + dif_index * 4;
    top_label[dif_index] = difficult;
    top_label[obj_index] = 1;
    // LOG(INFO) << "dif_index: " << dif_index << " class_label: " << class_label;
    top_label[class_index] = class_label;
    for (int j = 0; j < 4; ++j) {
      top_label[cor_index + j] = box_labels[i].box_[j];
    }
  }
}

INSTANTIATE_CLASS(BoxDataLayer);
REGISTER_LAYER_CLASS(BoxData);

}  // namespace caffe
