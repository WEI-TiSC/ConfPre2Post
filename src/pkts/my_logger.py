import logging
import coloredlogs


logger = logging.getLogger()
coloredlogs.install(level='DEBUG', logger=logger, format='%(asctime)s.%(msecs)03d [%(levelname)s] '
                                                         '[%(filename)s:%(lineno)d] %(message)s')

if __name__ == "__main__":
    logger.debug('this is a debug! message')
    logger.info('this is an info! message')
    logger.warning('this is a warning! message')
    logger.error('this is an error! message')
    logger.critical('this is a critical! message')

