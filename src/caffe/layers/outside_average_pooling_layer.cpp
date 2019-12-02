#include <algorithm>
#include <cfloat>
#include <vector>

#include "caffe/layers/outside_average_pooling_layer.hpp"

using std::max;
using std::min;
using std::floor;
using std::ceil;

namespace caffe {

template <typename Dtype>
void OutsideAveragePoolingLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  OutsideAveragePoolingParameter outside_average_pool_param = this->layer_param_.outside_average_pooling_param();
  spatial_scale_ = outside_average_pool_param.spatial_scale();
  LOG(INFO) << "Spatial scale: " << spatial_scale_;
}

template <typename Dtype>
void OutsideAveragePoolingLayer<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  channels_ = bottom[0]->channels();
  height_ = bottom[0]->height();
  width_ = bottom[0]->width();
  top[0]->Reshape(bottom[1]->num(), channels_, 1, 1);
}

template <typename Dtype>
void OutsideAveragePoolingLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
      NOT_IMPLEMENTED;
}

template <typename Dtype>
void OutsideAveragePoolingLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
      NOT_IMPLEMENTED;
}


#ifdef CPU_ONLY
STUB_GPU(OutsideAveragePoolingLayer);
#endif

INSTANTIATE_CLASS(OutsideAveragePoolingLayer);
REGISTER_LAYER_CLASS(OutsideAveragePooling);

}  // namespace caffe
