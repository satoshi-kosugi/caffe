#include <algorithm>
#include <cfloat>
#include <vector>

#include "caffe/layers/outside_average_pooling_layer.hpp"


using std::max;
using std::min;

namespace caffe {

template <typename Dtype>
__global__ void OutsideAveragePoolForward(const int nthreads, const Dtype* bottom_data,
    const Dtype spatial_scale, const int channels, const int height, const int width,
    const Dtype* bottom_rois, Dtype* top_data) {
  CUDA_KERNEL_LOOP(index, nthreads) {
    // (n, c, 1, 1) is an element in the pooled output
    int c = index % channels;
    int n = index / channels;

    bottom_rois += n * 5;
    int roi_batch_ind = bottom_rois[0];
    int roi_start_w = round(bottom_rois[1] * spatial_scale);
    int roi_start_h = round(bottom_rois[2] * spatial_scale);
    int roi_end_w = round(bottom_rois[3] * spatial_scale);
    int roi_end_h = round(bottom_rois[4] * spatial_scale);

    Dtype pooled_sum = 0;
    bottom_data += (roi_batch_ind * channels + c) * height * width;

    for (int h = roi_start_h; h <= roi_end_h; ++h) {
      for (int w = roi_start_w; w <= roi_end_w; ++w) {
        int bottom_index = h * width + w;
        pooled_sum += bottom_data[bottom_index];
      }
    }

    // int roi_width = roi_end_w - roi_start_w + 1;
    // int roi_height = roi_end_h - roi_start_h + 1;
    // int area = width * height - roi_width * roi_height;

    int area = width * height;

    if (area>0){
      top_data[index] = (top_data[index] - pooled_sum) / area;
    } else {
      top_data[index] = 0;
    }
  }
}

template <typename Dtype>
__global__ void GlobalSum(const int nthreads, const Dtype* bottom_data,
    const Dtype spatial_scale, const int num, const int channels,
    const int height, const int width,
    const Dtype* bottom_rois, Dtype* top_data) {
  CUDA_KERNEL_LOOP(index, nthreads) {
    // (n, c, 1, 1) is an element in the pooled output
    int c = index;

    Dtype pooled_sum = 0;
    bottom_data += (0 * channels + c) * height * width;

    for (int h = 0; h < height; ++h) {
      for (int w = 0; w < width; ++w) {
        int bottom_index = h * width + w;
        pooled_sum += bottom_data[bottom_index];
      }
    }

    for (int n = 0; n < num; ++n) {
      top_data[n*channels + c] = pooled_sum;
    }
  }
}

template <typename Dtype>
void OutsideAveragePoolingLayer<Dtype>::Forward_gpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  const Dtype* bottom_data = bottom[0]->gpu_data();
  const Dtype* bottom_rois = bottom[1]->gpu_data();
  Dtype* top_data = top[0]->mutable_gpu_data();
  int count = top[0]->count();
  int num = top[0]->num();
  // NOLINT_NEXT_LINE(whitespace/operators)
  GlobalSum<Dtype><<<CAFFE_GET_BLOCKS(channels_), CAFFE_CUDA_NUM_THREADS>>>(
      channels_, bottom_data, spatial_scale_, num, channels_, height_, width_,
      bottom_rois, top_data);
  CUDA_POST_KERNEL_CHECK;

  OutsideAveragePoolForward<Dtype><<<CAFFE_GET_BLOCKS(count), CAFFE_CUDA_NUM_THREADS>>>(
      count, bottom_data, spatial_scale_, channels_, height_, width_,
      bottom_rois, top_data);
  CUDA_POST_KERNEL_CHECK;
}

template <typename Dtype>
__global__ void OutsideAveragePoolBackward(const int nthreads, const Dtype* top_diff,
    const int num_rois, const Dtype spatial_scale,
    const int channels, const int height, const int width,
    Dtype* bottom_diff, const Dtype* bottom_rois) {
  CUDA_KERNEL_LOOP(index, nthreads) {
    // (n, c, h, w) coords in bottom data
    int w = index % width;
    int h = (index / width) % height;
    int c = (index / width / height) % channels;
    int n = index / width / height / channels;

    Dtype gradient = 0;
    // Accumulate gradient over all ROIs that pooled this element
    for (int roi_n = 0; roi_n < num_rois; ++roi_n) {
      const Dtype* offset_bottom_rois = bottom_rois + roi_n * 5;
      int roi_batch_ind = offset_bottom_rois[0];
      // Skip if ROI's batch index doesn't match n
      if (n != roi_batch_ind) {
        continue;
      }

      int roi_start_w = round(offset_bottom_rois[1] * spatial_scale);
      int roi_start_h = round(offset_bottom_rois[2] * spatial_scale);
      int roi_end_w = round(offset_bottom_rois[3] * spatial_scale);
      int roi_end_h = round(offset_bottom_rois[4] * spatial_scale);

      const bool out_roi = (w < roi_start_w || w > roi_end_w ||
                                   h < roi_start_h || h > roi_end_h);
      if (!out_roi) {
        continue;
      }

      // const bool in_roi = (w >= roi_start_w && w <= roi_end_w &&
      //                      h >= roi_start_h && h <= roi_end_h);
      // if (!in_roi) {
      //   continue;
      // }

      // int roi_width = roi_end_w - roi_start_w + 1;
      // int roi_height = roi_end_h - roi_start_h + 1;
      // int area = width * height - roi_width * roi_height;

      int area = width * height;

      int offset = roi_n * channels + c;
      const Dtype* offset_top_diff = top_diff + offset;

      gradient += offset_top_diff[0] / area;
    }
    bottom_diff[index] = gradient;
  }
}

template <typename Dtype>
void OutsideAveragePoolingLayer<Dtype>::Backward_gpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
  if (!propagate_down[0]) {
    return;
  }
  const Dtype* bottom_rois = bottom[1]->gpu_data();
  const Dtype* top_diff = top[0]->gpu_diff();
  Dtype* bottom_diff = bottom[0]->mutable_gpu_diff();
  const int count = bottom[0]->count();
  caffe_gpu_set(count, Dtype(0.), bottom_diff);
  // NOLINT_NEXT_LINE(whitespace/operators)
  OutsideAveragePoolBackward<Dtype><<<CAFFE_GET_BLOCKS(count), CAFFE_CUDA_NUM_THREADS>>>(
      count, top_diff, top[0]->num(), spatial_scale_, channels_,
      height_, width_, bottom_diff, bottom_rois);
  CUDA_POST_KERNEL_CHECK;
}

INSTANTIATE_LAYER_GPU_FUNCS(OutsideAveragePoolingLayer);

}  // namespace caffe
