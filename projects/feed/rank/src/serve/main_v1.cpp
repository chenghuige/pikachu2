#include <string>
#include <algorithm>
#include <sstream>
#include <iostream>
#include <fstream>
#include <vector>
#include <unordered_map>
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

using namespace tensorflow;
//using namespace std;

int main(int argc, char* argv[]) {
    // Initialize a tensorflow session
    tensorflow::SessionOptions options;
    tensorflow::ConfigProto& config = options.config;
    config.set_allow_soft_placement(true);

    Session* session;
    Status status = NewSession(options, &session);
    if (!status.ok()) {
        std::cerr << status.ToString() << "\n";
        return 1;
    } else {
        std::cout << "Session created successfully" << std::endl;
    }

    // Load graph protobuf
    GraphDef graph_def;
    std::string graph_path = argv[1];
    status = ReadBinaryProto(Env::Default(), graph_path, &graph_def);
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
    std::vector<int> v1({0,1,5,10,11, 13,15,29,26,28});
    std::vector<float> v2({0.1,0.2,0.5,0.9,0.1, 0.2,0.3,0.4,0.5,0.1});
    std::vector<int> v3({0,0,0,0,0, 1,1,1,1,1});
    int art_count = 2;

    int size = v1.size();
    std::cout << "tensor successfully 1" << std::endl;
    auto Placeholder_2 = test::AsTensor<int>(v1, {size});
    std::cout << "tensor successfully 2" << std::endl;
    auto Placeholder_3 = test::AsTensor<float>(v2, {size});
    std::cout << "tensor successfully 3" << std::endl;
    auto Placeholder_4 = test::AsTensor<int>(v3, {size});
    std::cout << "tensor successfully 4" << std::endl;
    std::vector<std::pair<std::string, Tensor> > inputs = {
        {"Placeholder_2", Placeholder_2},
        {"Placeholder_3", Placeholder_3},
        {"Placeholder_4", Placeholder_4},
    };

    std::vector<tensorflow::Tensor> outputs; 

    status = session->Run(inputs,{"Sigmoid"},{},&outputs);
    if (!status.ok()) {
        std::cerr << status.ToString() << std::endl;
        return 1;
    }
    else{
        std::cout << "Run session successfully" << std::endl;
    }
    if (outputs.size() == 0) {
      std::cout << "outputs.size() == 0" << std::endl;
    } else if (outputs[0].dtype() != tensorflow::DT_FLOAT) {
      std::cout << "dtype error" << std::endl;
    } else if (outputs[0].shape().dims() != 2) {
      std::cout << "dims error:" << outputs[0].shape().dims()  << std::endl;
    } else if (outputs[0].shape().dim_size(0) != art_count) {
      std::cout << "dim_size error:" << outputs[0].shape().dim_size(0) << "/" << outputs[0].shape().dim_size(1)  << std::endl;
    }

    //auto softmax = outputs[0].tensor<float,2>();

    std::cout << outputs[0].DebugString() << std::endl;
    //std::cout << "output value: " << softmax << std::endl;
    session->Close();

    return 0;

}
