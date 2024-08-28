#include <iostream>
#include <memory>

int main() {
    // 创建不同类型的智能指针
    std::shared_ptr<int> sharedPtr;
    std::unique_ptr<int> uniquePtr;
    std::weak_ptr<int> weakPtr;

    // 输出每种智能指针的 sizeof
    std::cout << "sizeof(std::shared_ptr<int>): " << sizeof(sharedPtr) << " bytes\n";
    std::cout << "sizeof(std::unique_ptr<int>): " << sizeof(uniquePtr) << " bytes\n";
    std::cout << "sizeof(std::weak_ptr<int>): " << sizeof(weakPtr) << " bytes\n";

    return 0;
}
