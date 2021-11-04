#include <string>
#include <algorithm>
#include <sstream>
#include <iostream>
#include <fstream>
#include <vector>
#include <unordered_map>
#include <sys/time.h>
//#include "util.h"
#include "tensorflow/core/public/session.h"
#include "tensorflow/core/platform/env.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/util/sparse/sparse_tensor.h"
#include "tensorflow/core/framework/tensor_testutil.h"
#include "tensorflow/core/framework/types.h"
#include "tensorflow/core/framework/tensor.pb.h"
#include "tensorflow/core/framework/variant_encode_decode.h"
#include "tensorflow/core/framework/variant_tensor_data.h"
#include "tensorflow/core/lib/strings/strcat.h"
#include "tensorflow/core/platform/logging.h"
#include "tensorflow/core/platform/test.h"
#include "tensorflow/core/platform/test_benchmark.h"
#include <tensorflow/cc/framework/scope.h>
#include <tensorflow/core/public/session.h>
#include <tensorflow/core/protobuf/meta_graph.pb.h>

#include "tensorflow/c/c_test_util.h"
//#include "tensorflow/c/tf_status.h"
#include "tensorflow/cc/saved_model/signature_constants.h"
#include "tensorflow/cc/saved_model/tag_constants.h"
//#include "tensorflow/core/example/example.pb.h"
//#include "tensorflow/core/example/feature.pb.h"
//#include "tensorflow/core/framework/api_def.pb.h"
#include "tensorflow/core/framework/common_shape_fns.h"
#include "tensorflow/core/framework/graph.pb.h"
#include "tensorflow/core/framework/kernel_def.pb.h"
#include "tensorflow/core/framework/node_def.pb.h"
#include "tensorflow/core/framework/node_def_util.h"
#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/op_def.pb.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/partial_tensor_shape.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/framework/tensor_shape.pb.h"
#include "tensorflow/core/framework/types.pb.h"
#include "tensorflow/core/graph/tensor_id.h"
#include "tensorflow/core/lib/core/status_test_util.h"
#include "tensorflow/core/lib/io/path.h"
#include "tensorflow/core/lib/strings/str_util.h"
#include "tensorflow/core/lib/strings/strcat.h"
#include "tensorflow/core/platform/test.h"
//#include "tensorflow/core/protobuf/error_codes.pb.h"
#include "tensorflow/core/protobuf/meta_graph.pb.h"
#include "tensorflow/core/util/equal_graph_def.h"

using namespace tensorflow;
//using namespace std;

namespace {
    class MyFileStream : public ::tensorflow::protobuf::io::ZeroCopyInputStream {
    public:
        explicit MyFileStream(RandomAccessFile* file) : file_(file), pos_(0) {}

        void BackUp(int count) override { pos_ -= count;  }
        bool Skip(int count) override {
            pos_ += count;
            return true;
        }
        protobuf_int64 ByteCount() const override { return pos_;  }
        Status status() const { return status_;  }

        bool Next(const void** data, int* size) override {
            StringPiece result;
            Status s = file_->Read(pos_, kBufSize, &result, scratch_);
            if (result.empty()) {
                status_ = s;
                return false;                            
            }
            pos_ += result.size();
            *data = result.data();
            *size = result.size();
            return true;                               
        }

    private:
        static const int kBufSize = 512 << 10;

        RandomAccessFile* file_;
        int64 pos_;
        Status status_;
        char scratch_[kBufSize];
    };
}

Status MyReadBinaryProto(Env* env, const std::string& fname,
                               ::tensorflow::protobuf::MessageLite* proto) {
    std::unique_ptr<RandomAccessFile> file;
    TF_RETURN_IF_ERROR(env->NewRandomAccessFile(fname, &file));
    std::unique_ptr<MyFileStream> stream(new MyFileStream(file.get()));

    // TODO(jiayq): the following coded stream is for debugging purposes to allow
    // one to parse arbitrarily large messages for MessageLite. One most likely
    // doesn't want to put protobufs larger than 64MB on Android, so we should
    // eventually remove this and quit loud when a large protobuf is passed in.
    ::tensorflow::protobuf::io::CodedInputStream coded_stream(stream.get());
    // Total bytes hard limit / warning limit are set to 2GB and 1GB
    // respectively.
    coded_stream.SetTotalBytesLimit(INT_MAX, 1024LL << 20);
    //coded_stream.SetTotalBytesLimit(1024LL << 20, 1024LL << 20);

    if (!proto->ParseFromCodedStream(&coded_stream) ||
            !coded_stream.ConsumedEntireMessage()) {
        TF_RETURN_IF_ERROR(stream->status());
        return errors::DataLoss("Can't parse ", fname, " as binary proto");
    }
    return Status::OK();
}

int main(int argc, char* argv[]) {
    // Initialize a tensorflow session
    tensorflow::SessionOptions options;
    tensorflow::ConfigProto& config = options.config;
    config.set_allow_soft_placement(true);

    Session* session;
    Status status = NewSession(options, &session);
    
    //TF_Status* status_ = TF_NewStatus();
    //TF_LoadLibrary("/home/gezi/mine/pikachu/projects/feed/rank/src/ops/time.so", status_);
    
   if (!status.ok()) {
        std::cerr << status.ToString() << "\n";
        return 1;
    } else {
        std::cout << "Session created successfully" << std::endl;
    }

    // Load graph protobuf
    GraphDef graph_def;
    std::string graph_path = argv[1];
    //status = ReadBinaryProto(Env::Default(), graph_path, &graph_def);
    status = MyReadBinaryProto(Env::Default(), graph_path, &graph_def);
    if (!status.ok()) {
        std::cerr << status.ToString() << std::endl;
        return 1;
    } else {
        std::cout << "Load graph protobuf successfully" << std::endl;
    }

    // Add the graph to the session
    status = session->Create(graph_def);
    if (!status.ok()) {
        std::cerr << status.ToString() << std::endl;
        return 1;
    } else {
        std::cout << "Add graph to session successfully" << std::endl;
    }
    for (int i = 0; i < graph_def.node_size(); ++i) {
        auto& node = graph_def.node(i);
        //node.PrintDebugString();
    }
    // Setup inputs and outputs
    std::vector<int64> v1({256, 298, 540238, 779534, 1018811, 1258224, 1265534, 1265643, 1265654, 1265666, 1265688, 1265711, 1303442, 1329553, 1345794, 1351339, 1357877, 1383523, 1383640, 1391890, 1417815, 1417939, 1420057, 1420067, 1420093, 1420124, 1420140, 1689212, 1778231, 1912066, 2037414, 2425736, 2426845, 2426849, 2426965, 2427040, 2546659, 2596769, 2662528, 2670288, 2671387, 2700036, 2703910, 2708492, 2709243, 2789953, 2804355, 2822478, 2825661, 2892795, 2902912, 2904485, 2907331, 2907424, 2907570, 2907676, 2907734
            , 256, 298, 540238, 779534, 1018811, 125824, 1265534, 1265643, 1265654, 1265666, 1265688, 1265711, 1303442, 1329553, 1345794, 1351339, 1357877, 1383523, 1383640, 1391890, 1417815, 1417939, 1420057, 1420067, 1420093, 1420124, 1420140, 1689212, 1778231, 1912066, 2037414, 2425736, 2426845, 2426849, 2426965, 2427040, 2546659, 2596769, 2662528, 2670288, 2671387, 2700036, 2703910, 2708392, 2709243, 2789953, 2804355, 2822478, 2825661, 2892785, 2902912, 2904485, 2907331, 2907424, 2907570, 2907676, 290734
            , 257, 298, 540238, 779534, 1018811, 125824, 1265534, 1265643, 1265654, 1265666, 1265688, 1265711, 1303442, 1329553, 1345794, 1351339, 1357877, 1383523, 1383640, 1391890, 1417815, 1417939, 1420057, 1420067, 1420093, 1420124, 1420140, 1689212, 1778231, 1912066, 2037414, 2425736, 2426845, 2436849, 2426965, 2427140, 2546659, 2596769, 2662528, 2670288, 2671387, 2700036, 2703910, 2708392, 2709243, 2789953, 2804355, 2822478, 2823661, 2892785, 2902912, 2904485, 2907331, 2907424, 2907570, 2907676, 290734});
    std::vector<float> v2({1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.564064, 0.057216, 0.03559, 0.012151, 0.047539, 0.093495, 0.050457, 1.0, 1.0, 0.3, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.0072500003, 1.028952654469, 0.99145, 1.02895, 0.903, 0.0072500003, 0.99145, 1.02895, 0.903, 0.0072500003, 0.99145, 1.02895, 0.903, 0.99145, 1.02895, 0.903, 1.0, 1.0, 1.0, 1.0, 1.0
            , 1.0, 1.0, 1.0, 0.0123, 0.234, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.564064, 0.057216, 0.03559, 0.012151, 0.047539, 0.093495, 0.050457, 1.0, 1.0, 0.3, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.0072500003, 1.028952654469, 0.99145, 1.02895, 0.903, 0.0072500003, 0.99145, 1.02895, 0.903, 0.0072500003, 0.99145, 1.02895, 0.903, 0.99145, 1.02895, 0.903, 1.0, 1.0, 0.08, 0.01, 1.0
            , 1.0, 1.0, 1.0, 0.0123, 0.234, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.564064, 0.057216, 0.03559, 0.012151, 0.047539, 0.093495, 0.050457, 1.0, 1.0, 0.3, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.0072500003, 1.0289654469, 0.99145, 1.02895, 0.903, 0.0172500003, 0.9145, 1.02895, 0.903, 0.007200003, 0.9145, 1.02895, 0.903, 0.99145, 1.02895, 0.903, 1.0, 1.0, 0.08, 0.102, 1.0});
    std::vector<int64> v3({1, 2, 5, 6, 7, 8, 9, 10, 11, 12, 13, 15, 16, 17, 18, 18, 18, 19, 19, 19, 19, 19, 20, 21, 22, 23, 24, 26, 27, 28, 29, 35, 36, 37, 38, 39, 41, 42, 45, 46, 47, 49, 49, 50, 51, 53, 53, 54, 55, 57, 58, 59, 64, 65, 66, 67, 68
            , 1, 2, 5, 6, 7, 8, 9, 10, 11, 12, 13, 15, 16, 17, 18, 18, 18, 19, 19, 19, 19, 19, 20, 21, 22, 23, 24, 26, 27, 28, 29, 35, 36, 37, 38, 39, 41, 42, 45, 46, 47, 48, 49, 50, 51, 53, 53, 54, 55, 57, 58, 59, 64, 65, 66, 67, 68
            , 1, 2, 5, 6, 7, 8, 9, 10, 11, 12, 13, 15, 16, 17, 18, 18, 18, 19, 19, 19, 19, 19, 20, 21, 22, 23, 24, 26, 27, 28, 29, 35, 36, 37, 38, 39, 41, 42, 45, 46, 47, 48, 49, 50, 51, 53, 53, 54, 55, 57, 58, 59, 64, 64, 66, 67, 68});
    for (auto& elem: v1) if (elem > 2400000) elem-=1000000;
    int art_count = 3;

    int size = v1.size();
    std::cout << "tensor successfully 1" << std::endl;
    auto Placeholder_2 = test::AsTensor<int64>(v1, {art_count, size/art_count});
    std::cout << "tensor successfully 2" << std::endl;
    auto Placeholder_3 = test::AsTensor<float>(v2, {art_count, size/art_count});
    std::cout << "tensor successfully 3" << std::endl;
    auto Placeholder_4 = test::AsTensor<int64>(v3, {art_count, size/art_count});
    std::cout << "tensor successfully 4" << std::endl;
    std::vector<std::pair<std::string, Tensor> > inputs = {
        {"index_feed", Placeholder_2},
        {"value_feed", Placeholder_3},
        {"field_feed", Placeholder_4},
    };
    
    std::cout << "input show:" << std::endl;
    for (auto& pair: inputs) {
        std::cout << pair.first << ":" << pair.second.DebugString() << std::endl;
    }
    std::vector<tensorflow::Tensor> outputs;
    // status = session->Run({}, {}, {"init_all_tables"}, nullptr);
    // if (!status.ok()) {
    //     std::cerr << status.ToString() << std::endl;
    //     return 1;
    // } else{
    //     std::cout << "Session init_all_tables ok" << std::endl;
    // }
    status = session->Run(inputs,{"pred"},{},&outputs);
    std::cout << "finish!" << std::endl;
    if (!status.ok()) {
        std::cerr << status.ToString() << std::endl;
        return 1;
    } else{
        std::cout << "Run session successfully" << std::endl;
    }

    struct timeval tv;
    gettimeofday(&tv,NULL);
    long long start = (long long) tv.tv_sec * 1000L + tv.tv_usec / 1000;
    for (int i = 0; i < 10; ++i) {
        std::cout << i << std::endl;
        auto Placeholder_2 = test::AsTensor<int64>(v1, {art_count, size/art_count});
        auto Placeholder_3 = test::AsTensor<float>(v2, {art_count, size/art_count});
        auto Placeholder_4 = test::AsTensor<int64>(v3, {art_count, size/art_count});
        std::vector<std::pair<std::string, Tensor> > inputs = {
            {"index_feed", Placeholder_2},
            {"value_feed", Placeholder_3},
            {"field_feed", Placeholder_4},
        };
        std::vector<tensorflow::Tensor> outputs_tmp;
        status = session->Run(inputs,{"pred"},{},&outputs_tmp);
    }
    gettimeofday(&tv,NULL);
    long long end = (long long) tv.tv_sec * 1000L + tv.tv_usec / 1000;
    std::cout << "Time:" << end - start << std::endl;

    if (outputs.size() == 0) {
      std::cout << "outputs.size() == 0" << std::endl;
    } else if (outputs[0].dtype() != tensorflow::DT_FLOAT) {
      std::cout << "dtype error" << std::endl;
    } else if (outputs[0].shape().dims() != 1) {
      std::cout << "dims error:" << outputs[0].shape().dims()  << std::endl;
    } else if (outputs[0].shape().dim_size(0) != art_count) {
      std::cout << "dim_size error:" << outputs[0].shape().dim_size(0) << std::endl;
    }

    //auto softmax = outputs[0].tensor<float,2>();

    std::cout << outputs[0].DebugString() << std::endl;
    //std::cout << "output value: " << softmax << std::endl;
    session->Close();

    return 0;

}
