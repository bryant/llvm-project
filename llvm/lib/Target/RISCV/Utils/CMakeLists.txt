set(LLVM_LINK_COMPONENTS
  Support
  )

add_llvm_library(LLVMRISCVUtils
  RISCVBaseInfo.cpp
  RISCVMatInt.cpp
  )
