// Auto-generated macro to instanciate and initialize opcode resolver based on TFLite flatbuffers in config directory
#ifndef SL_TFLITE_MICRO_OPCODE_RESOLVER_H
#define SL_TFLITE_MICRO_OPCODE_RESOLVER_H

#define SL_TFLITE_MICRO_OPCODE_RESOLVER(opcode_resolver) \
static tflite::MicroMutableOpResolver<7> opcode_resolver; \
opcode_resolver.AddConv2D(); \
opcode_resolver.AddDepthwiseConv2D(); \
opcode_resolver.AddAdd(); \
opcode_resolver.AddAveragePool2D(); \
opcode_resolver.AddReshape(); \
opcode_resolver.AddFullyConnected(); \
opcode_resolver.AddSoftmax(); \


#endif // SL_TFLITE_MICRO_OPCODE_RESOLVER_H
