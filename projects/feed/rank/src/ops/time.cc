#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/shape_inference.h"

#include <ctime>
#include <iostream>

#include <boost/date_time/gregorian/gregorian.hpp>
#include <boost/date_time/posix_time/posix_time.hpp>
#include <boost/date_time/local_time_adjustor.hpp>
#include <boost/date_time/c_local_time_adjustor.hpp>
  
using namespace tensorflow;
using std::vector;

REGISTER_OP("Time")
.Attr("time_bins_per_hour: int = 6")
.Input("stamp: int64")
.Output("z: int64")
.SetIsCommutative()
.SetShapeFn([](::tensorflow::shape_inference::InferenceContext* c) {
      c->set_output(0, c->input(0));
      return Status::OK();
    })
.Doc(R"doc(
Given timestamps output it's buckets value
)doc");

typedef long long int64;

class TimeOp : public OpKernel {

  public:

    explicit TimeOp(OpKernelConstruction* context) : OpKernel(context) {
      OP_REQUIRES_OK(context,
                     context->GetAttr("time_bins_per_hour", &time_bins_per_hour_));
    }

    void Compute(OpKernelContext* context) override {
      const Tensor& input_tensor = context->input(0);
      auto timestamps = input_tensor.flat<int64>(); 

      Tensor* output = NULL;
      OP_REQUIRES_OK(context,
                    context->allocate_output(0, input_tensor.shape(), &output));
      auto output_flat = output->template flat<int64>();

      for (size_t i = 0; i < timestamps.size(); i++)
      {
        int64 timestamp = timestamps(i);
        int64 val = 0;
        if (timestamp > 0)
        {
          boost::posix_time::ptime pt = boost::posix_time::from_time_t(timestamp);
          typedef boost::date_time::c_local_adjustor<boost::posix_time::ptime> local_adj;
          pt = local_adj::utc_to_local(pt);
          tm t = to_tm(pt);
          int span = int(60 / time_bins_per_hour_);
          val = int64(t.tm_hour * time_bins_per_hour_ + int(t.tm_min / span) + 1);
        }
        output_flat(i) = int64(val + 1);
      }
    }
    
  private:
    int time_bins_per_hour_ = 6;
};

REGISTER_OP("Weekday")
.Input("stamp: int64")
.Output("z: int64")
.SetIsCommutative()
.SetShapeFn([](::tensorflow::shape_inference::InferenceContext* c) {
      c->set_output(0, c->input(0));
      return Status::OK();
    })
.Doc(R"doc(
Given timestamps output weekday (+2)
)doc");

class WeekdayOp : public OpKernel {

  public:

    explicit WeekdayOp(OpKernelConstruction* context) : OpKernel(context) {}

    void Compute(OpKernelContext* context) override {
      const Tensor& input_tensor = context->input(0);
      auto timestamps = input_tensor.flat<int64>(); 

      Tensor* output = NULL;
      OP_REQUIRES_OK(context,
                    context->allocate_output(0, input_tensor.shape(), &output));
      auto output_flat = output->template flat<int64>();

      for (size_t i = 0; i < timestamps.size(); i++)
      {
        int64 timestamp = timestamps(i);
        int64 val = 0;
        if (timestamp > 0)
        {
          boost::posix_time::ptime pt = boost::posix_time::from_time_t(timestamp);
          typedef boost::date_time::c_local_adjustor<boost::posix_time::ptime> local_adj;
          pt = local_adj::utc_to_local(pt);
          val = int64(pt.date().day_of_week() + 1);
        }
        output_flat(i) = int64(val + 1);
      }
    }
    
};

REGISTER_OP("Timespan")
.Input("impress: int64")
.Input("pagetime: int64")
.Output("z: int64")
.SetIsCommutative()
.SetShapeFn([](::tensorflow::shape_inference::InferenceContext* c) {
      c->set_output(0, c->input(0));
      return Status::OK();
    })
.Doc(R"doc(
Given timestamps output weekday (+2)
)doc");

class TimespanOp : public OpKernel {

  public:

    explicit TimespanOp(OpKernelConstruction* context) : OpKernel(context) {}

    void Compute(OpKernelContext* context) override {
      const Tensor& input_tensor = context->input(0);
      const Tensor& input_tensor2 = context->input(1);
      auto impress = input_tensor.flat<int64>(); 
      auto pt = input_tensor2.flat<int64>();

      Tensor* output = NULL;
      OP_REQUIRES_OK(context,
                    context->allocate_output(0, input_tensor.shape(), &output));

      auto output_flat = output->template flat<int64>();

      for (size_t i = 0; i < impress.size(); i++)
      {
        int64 a  = impress(i);
        int64 b = pt(i);
        int64 val = 0;
        if (a > 0 && b > 0 && a > b)
        {
          val = int(std::log2(a - b) * 5) + 1;
          if (val > 200)
          {
            val = 200;
          }
        }
        output_flat(i) = int64(val + 1);
      }
    }  
};


REGISTER_KERNEL_BUILDER(Name("Time").Device(DEVICE_CPU), TimeOp);
REGISTER_KERNEL_BUILDER(Name("Weekday").Device(DEVICE_CPU), WeekdayOp);
REGISTER_KERNEL_BUILDER(Name("Timespan").Device(DEVICE_CPU), TimespanOp);
