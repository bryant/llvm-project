set(LLVM_LINK_COMPONENTS
  Support
  )

add_llvm_library(LLVMX86Utils
  X86ShuffleDecode.cpp
  )
