#include "caffe/layers/multi_class_cross_entropy_loss_layer.hpp"
namespace caffe {
template <typename Dtype>
__global__ void MulticlassCrossEntropyLossForwardGPU(const int nthreads,
          const Dtype* input_data, const Dtype* label, Dtype* loss) {
  CUDA_KERNEL_LOOP(index, nthreads) {
    Dtype tmp = input_data[index];
    if (int(label[index]) == 0) {
      Dtype tmp_inverse = 1 - tmp;
      if (tmp_inverse < 1e-10) {
        tmp_inverse = 1e-10;
      }
      loss[index] = -log(tmp_inverse);
      // loss[index] = -log(1 - tmp);
    }
    else {
      if (tmp < 1e-10) {
        tmp = 1e-10;
      }
      loss[index] = -log(tmp);
    }
  }
}
template <typename Dtype>
void MulticlassCrossEntropyLossLayer<Dtype>::Forward_gpu(
    const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
  const Dtype* input_data = bottom[0]->gpu_data();
  const Dtype* label = bottom[1]->gpu_data();
  const int nthreads = batch_size * channels;
  // Since this memory is not used for anything until it is overwritten
  // on the backward pass, we use it here to avoid having to allocate new GPU
  // memory to accumulate intermediate results in the kernel.
  Dtype* loss_data = bottom[0]->mutable_gpu_diff();
  // Similarly, this memory is never used elsewhere, and thus we can use it
  // to avoid having to allocate additional GPU memory.
  // NOLINT_NEXT_LINE(whitespace/operators)
  MulticlassCrossEntropyLossForwardGPU<Dtype><<<CAFFE_GET_BLOCKS(nthreads),
      CAFFE_CUDA_NUM_THREADS>>>(nthreads, input_data, label, loss_data);
  Dtype loss;
  caffe_gpu_asum(nthreads, loss_data, &loss);
  loss /= batch_size;
  top[0]->mutable_cpu_data()[0] = loss;
}
template <typename Dtype>
__global__ void MulticlassCrossEntropyLossBackwardGPU(const int nthreads,
          const Dtype* input_data, const Dtype* label, Dtype* bottom_diff,
                  const int channels) {
  CUDA_KERNEL_LOOP(index, nthreads) {
    Dtype tmp = input_data[index];
    if (int(label[index]) == 0) {
      Dtype tmp_inverse = 1 - tmp;
      if (tmp_inverse < 1e-10) {
        tmp_inverse = 1e-10;
      }
      bottom_diff[index] = 1.0 / (channels * (tmp_inverse));
    }
    else {
      if (tmp < 1e-10) {
        tmp = 1e-10;
      }
      bottom_diff[index] = -1.0 / (channels * tmp);
    }
  }
}
template <typename Dtype>
void MulticlassCrossEntropyLossLayer<Dtype>::Backward_gpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
  if (propagate_down[1]) {
    LOG(FATAL) << this->type()
               << " Layer cannot backpropagate to label inputs.";
  }
  if (propagate_down[0]) {
    Dtype* bottom_diff = bottom[0]->mutable_gpu_diff();
    const Dtype* input_data = bottom[0]->gpu_data();
    const Dtype* label = bottom[1]->gpu_data();
    const int nthreads = batch_size * channels;

    MulticlassCrossEntropyLossBackwardGPU<Dtype><<<CAFFE_GET_BLOCKS(nthreads),
        CAFFE_CUDA_NUM_THREADS>>>(nthreads, input_data, label, bottom_diff,
        channels);
    const Dtype loss_weight = top[0]->cpu_diff()[0];

    caffe_gpu_scal(bottom[0]->count(), loss_weight / batch_size, bottom_diff);
  }
}
INSTANTIATE_LAYER_GPU_FUNCS(MulticlassCrossEntropyLossLayer);
}  // namespace caffe
