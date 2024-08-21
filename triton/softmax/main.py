import unittest
from tests.test_softmax import test_softmax
from tests.test_performance import test_performance

def main():
    # 创建测试套件
    suite = unittest.TestSuite()
    
    # 添加测试用例
    suite.addTest(unittest.FunctionTestCase(test_softmax))
    suite.addTest(unittest.FunctionTestCase(test_performance))
    
    # 运行测试
    runner = unittest.TextTestRunner()
    runner.run(suite)

if __name__ == "__main__":
    main()
