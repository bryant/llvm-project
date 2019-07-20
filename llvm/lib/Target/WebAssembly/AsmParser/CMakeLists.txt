set(LLVM_LINK_COMPONENTS
  MC
  MCParser
  Support
  WebAssemblyInfo
  )

add_llvm_library(LLVMWebAssemblyAsmParser
  WebAssemblyAsmParser.cpp
  )
