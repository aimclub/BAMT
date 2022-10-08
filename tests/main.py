from NetworksTest import TestHybridBN, TestContinuousBN, TestDiscreteBN

import logging
import time

logging.getLogger('preprocessor').setLevel(logging.ERROR)

if __name__ == "__main__":
    t0 = time.time()
    dir = r"../data/real data/hack_processed_with_rf.csv"

    tests = [TestHybridBN(directory=dir),
             TestDiscreteBN(directory=dir),
             TestContinuousBN(directory=dir)]
    for test in tests:
        try:
            test.apply()
        except Exception as ex:
            print(ex)
            continue
    print(f"Total time: {time.time() - t0}")
