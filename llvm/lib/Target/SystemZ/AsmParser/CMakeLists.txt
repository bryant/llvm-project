set(LLVM_LINK_COMPONENTS
  MC
  MCParser
  Support
  SystemZDesc
  SystemZInfo
  )

add_llvm_library(LLVMSystemZAsmParser
  SystemZAsmParser.cpp
  )
