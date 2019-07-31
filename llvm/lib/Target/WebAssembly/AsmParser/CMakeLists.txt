add_llvm_library(LLVMWebAssemblyAsmParser
  WebAssemblyAsmParser.cpp

  LINK_COMPONENTS

  MC
  MCParser
  Support
  WebAssemblyInfo
  )
