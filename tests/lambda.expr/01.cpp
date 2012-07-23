// RUN: clang++ -c -emit-llvm -DCPPAMP -std=c++amp 01.cpp
#include <iostream>

int main() {
  int a = 1;
  int b = 2;
  int c;
  [=, &c] ()
#ifdef CPPAMP
    restrict(cpu)
#endif
    { c = a + b; } ();
  std::cout << "c = " << c << '\n';
  return 0;
}

