"""
Convert TF2 SavedModel to frozen graph (.pb) and text graph (.pbtxt).
This is supposed to be for OpenCV DNN module compatibility but
currently not working as it contains unsupported operations.
"""

import sys

import tensorflow as tf
from tensorflow.python.framework.convert_to_constants import (
    convert_variables_to_constants_v2,
)


def convert_savedmodel_to_frozen(saved_model_dir, output_pb_path):
    """
    Convert TF2 SavedModel to frozen graph
    """
    print(f"Loading SavedModel from: {saved_model_dir}")

    # Load the SavedModel
    model = tf.saved_model.load(saved_model_dir)

    # Get the concrete function
    concrete_func = model.signatures[tf.saved_model.DEFAULT_SERVING_SIGNATURE_DEF_KEY]

    # Convert variables to constants
    frozen_func = convert_variables_to_constants_v2(concrete_func)

    # Get the frozen graph
    frozen_graph = frozen_func.graph.as_graph_def()

    # Print input/output info
    print("\nInput tensors:")
    for input_tensor in concrete_func.inputs:
        print(
            f"  {input_tensor.name}, shape: {input_tensor.shape}, dtype: {input_tensor.dtype}"
        )

    print("\nOutput tensors:")
    for output_tensor in concrete_func.outputs:
        print(
            f"  {output_tensor.name}, shape: {output_tensor.shape}, dtype: {output_tensor.dtype}"
        )

    # Save frozen graph
    tf.io.write_graph(frozen_graph, ".", output_pb_path, as_text=False)
    print(f"\nFrozen graph saved to: {output_pb_path}")

    # Also save as text format (.pbtxt) for OpenCV
    output_pbtxt_path = output_pb_path.replace(".pb", ".pbtxt")
    tf.io.write_graph(frozen_graph, ".", output_pbtxt_path, as_text=True)
    print(f"Text graph saved to: {output_pbtxt_path}")


if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python convert_to_frozen.py <savedmodel_dir> <output.pb>")
        print(
            "Example: python convert_to_frozen.py ./models/saved_model ./models/frozen_model.pb"
        )
        sys.exit(1)

    saved_model_dir = sys.argv[1]
    output_pb = sys.argv[2]

    convert_savedmodel_to_frozen(saved_model_dir, output_pb)
