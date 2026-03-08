import onnx
from onnxruntime.quantization import quantize_dynamic, QuantType

# Definição das variáveis (garanta que elas existam antes da chamada)
model_fp32 = "facenet.onnx"
model_int8 = "facenet_int8.onnx"

quantize_dynamic(
    model_input=model_fp32,   
    model_output=model_int8,
    weight_type=QuantType.QUInt8
)

print(f"Modelo quantizado com sucesso: {model_int8}")