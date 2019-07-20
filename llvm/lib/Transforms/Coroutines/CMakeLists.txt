set(LLVM_LINK_COMPONENTS
  Analysis
  Core
  IPO
  ScalarOpts
  Support
  TransformUtils
  )

add_llvm_library(LLVMCoroutines
  Coroutines.cpp
  CoroCleanup.cpp
  CoroEarly.cpp
  CoroElide.cpp
  CoroFrame.cpp
  CoroSplit.cpp

  DEPENDS
  intrinsics_gen
  )
