add_llvm_library(LLVMRISCVUtils
  RISCVBaseInfo.cpp
  RISCVMatInt.cpp

  LINK_COMPONENTS

  Support
  )
