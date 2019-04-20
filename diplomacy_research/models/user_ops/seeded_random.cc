/** SeededRandom Op

    - Given a vector of seeds, generate a matrix of size (nb_seeds, size) where each row with identical seed is
      identical. (e.g. seeds of [10, 5, 10, 5] will have the same 1st and 3rd row, and the same 2nd and 4th row).
    - Rows with a seeds of 0, with use the graph seed, then the op seed, then a random seed.

    Attributes:
        seed: graph_seed. - The graph seed (defaults to 0)
        seed2: op_seed - The seed at the op construction (default to 0)

    Inputs:
        seeds (int32): vector of seeds. Batch of seeds used to generate random numbers. Vector length is batch size.
        offset (int32): integer to add to seeds as deterministic mask at initialization.
        size (int32): output size. Number of values to generate for each seed.

    Output: Matrix of generated random numbers, with shape (batch size, size).
**/
#include <vector>
#include <random>
#include <chrono>
#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/tensor_shape.h"
#include "tensorflow/core/framework/shape_inference.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/function.h"

#define M 2147483647U
#define A 2147483629U
#define C 2147483587U

using namespace tensorflow;

REGISTER_OP("SeededRandom")
    .Input("seeds: int32")
    .Input("offset: int32")
    .Input("size: int32")
    .SetIsStateful()                // seems necessary to force re-computation even when all inputs are constant.
    .Output("output: float")        // output format: (batch_size, size)
    .Attr("seed: int = 0")
    .Attr("seed2: int = 0")
    .SetShapeFn([](::tensorflow::shape_inference::InferenceContext* c) {
        shape_inference::ShapeHandle seeds_shape;
        shape_inference::ShapeHandle size_shape;
        TF_RETURN_IF_ERROR(c->WithRank(c->input(0), 1, &seeds_shape));
        TF_RETURN_IF_ERROR(c->WithRank(c->input(2), 0, &size_shape));

        size_t output_size = 0;
        const Tensor* size_tensor = c->input_tensor(2);
        if (size_tensor) { output_size = size_tensor->scalar<int32>()(); }

        std::vector<shape_inference::DimensionHandle> outputDimensions;
        outputDimensions.push_back(c->Dim(seeds_shape, 0));                                         // Batch size.
        outputDimensions.push_back(output_size ? c->MakeDim(output_size) : c->UnknownDim());        // Output size.
        c->set_output(0, c->MakeShape(outputDimensions));
        return Status::OK();
    });

class SeededRandomOp: public OpKernel {
    int _seed;
    int _seed2;
public:
    explicit SeededRandomOp(OpKernelConstruction* context): OpKernel(context) {
        OP_REQUIRES_OK(context, context->GetAttr("seed", &_seed));
        OP_REQUIRES_OK(context, context->GetAttr("seed2", &_seed2));
    }
    void Compute(OpKernelContext* context) override {
        // Get tensors.
        const Tensor& seeds_tensor = context->input(0);
        const Tensor& offset_tensor = context->input(1);
        const Tensor& size_tensor = context->input(2);

        // Get tensor shapes.
        const TensorShape& seeds_shape = seeds_tensor.shape();
        const TensorShape& offset_shape = offset_tensor.shape();
        const TensorShape& size_shape = size_tensor.shape();

        // Check inputs shapes .
        DCHECK_EQ(seeds_shape.dims(), 1);
        DCHECK_EQ(offset_shape.dims(), 0);
        DCHECK_EQ(size_shape.dims(), 0);

        // Get inputs data.
        auto seeds = seeds_tensor.vec<int32>();
        auto offset = offset_tensor.scalar<int32>()();
        auto output_size = size_tensor.scalar<int32>()();
        size_t batch_size = seeds_shape.dim_size(0);

        // Allocate output matrix (shape_prod, batch size).
        TensorShape output_shape;
        output_shape.AddDim(batch_size);
        output_shape.AddDim(output_size);
        Tensor* output = NULL;
        OP_REQUIRES_OK(context, context->allocate_output(0, output_shape, &output));
        auto output_matrix = output->matrix<float>();

        // Generate alternative seeds.
        std::vector<int32> alt_seeds(batch_size, 0);
        std::vector<int32> rng(batch_size, 0);
        if (_seed || _seed2) {
            auto seed = _seed ? _seed : _seed2;
            for (size_t i = 0; i < batch_size; ++i) { alt_seeds[i] = seed; }
        } else {
            std::default_random_engine generator(std::chrono::system_clock::now().time_since_epoch().count());
            std::uniform_int_distribution<int32> distribution(0, M - 1);
            for (size_t i = 0; i < batch_size; ++i) { alt_seeds[i] = distribution(generator); }
        }

        // Initialize RNG.
        for (size_t i = 0; i < batch_size; ++i) {
            rng[i] = ((seeds(i) ? seeds(i) : alt_seeds[i]) + offset) % M;
        }

        // Update RNG and generate output.
        for (size_t i = 0; i < batch_size; ++i) {
            for (size_t j = 0; j < output_size; ++j) {
                rng[i] = (A * rng[i] + C) % M;
                output_matrix(i, j) = float(rng[i]) / M;
            }
        }
    }
};

REGISTER_KERNEL_BUILDER(Name("SeededRandom").Device(DEVICE_CPU), SeededRandomOp);
REGISTER_OP_NO_GRADIENT("SeededRandom");
