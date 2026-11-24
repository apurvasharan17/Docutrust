from docutrust.metrics.ece import expected_calibration_error


def test_perfect_calibration():
    
    conf = [1.0] * 10
    correct = [True] * 10
    assert expected_calibration_error(conf, correct) == 0.0


def test_max_miscalibration():
    
    conf = [1.0] * 10
    correct = [False] * 10
    assert expected_calibration_error(conf, correct) == 1.0