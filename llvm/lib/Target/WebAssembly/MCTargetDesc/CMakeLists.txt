set(LLVM_LINK_COMPONENTS
  MC
  Support
  WebAssemblyInfo
  )

add_llvm_library(LLVMWebAssemblyDesc
  WebAssemblyAsmBackend.cpp
  WebAssemblyInstPrinter.cpp
  WebAssemblyMCAsmInfo.cpp
  WebAssemblyMCCodeEmitter.cpp
  WebAssemblyMCTargetDesc.cpp
  WebAssemblyTargetStreamer.cpp
  WebAssemblyWasmObjectWriter.cpp
)
