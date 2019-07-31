add_llvm_library(LLVMWebAssemblyDisassembler
  WebAssemblyDisassembler.cpp

  LINK_COMPONENTS

  MCDisassembler
  Support
  WebAssemblyDesc
  WebAssemblyInfo
  )
