import os
import logging
import time


def create_logger(output_path, cfg_name):
    # set up logger
    if not os.path.exists(output_path):
        os.makedirs(output_path)
    assert os.path.exists(output_path), '{} does not exist'.format(output_path)

    log_file = '{}_{}.log'.format(cfg_name, time.strftime('%Y-%m-%d-%H-%M'))
    head = '%(asctime)-15s %(message)s'
    logging.basicConfig(filename=os.path.join(output_path, log_file), format=head)
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)

    return logger


if __name__ == '__main__':
    test_logger = create_logger('./temp', 'xx')
    test_logger.info('Info.....')
    test_logger.critical('Critical....')
    test_logger.warning('Warning....')