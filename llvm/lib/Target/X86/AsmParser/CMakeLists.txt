set(LLVM_LINK_COMPONENTS
  MC
  MCParser
  Support
  X86Desc
  X86Info
  )

add_llvm_library(LLVMX86AsmParser
  X86AsmParser.cpp
  )
