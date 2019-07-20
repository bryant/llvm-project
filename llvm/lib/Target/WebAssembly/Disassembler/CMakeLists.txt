set(LLVM_LINK_COMPONENTS
  MCDisassembler
  Support
  WebAssemblyDesc
  WebAssemblyInfo
  )

add_llvm_library(LLVMWebAssemblyDisassembler
  WebAssemblyDisassembler.cpp
  )
