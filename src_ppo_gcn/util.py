import logging
import sys, os
import subprocess
import numpy as np
import warnings
warnings.filterwarnings("ignore", message="numpy.dtype size changed")
warnings.filterwarnings("ignore", message="numpy.ufunc size changed")

log = logging.getLogger(__name__)

def initialise_logging(log_file,debug=False):

	file_handler = logging.FileHandler(filename=log_file)
	stdout_handler = logging.StreamHandler(sys.stdout)
	handlers = [file_handler, stdout_handler]

	log_level = logging.INFO
	if debug:
		log_level = logging.DEBUG

	logging.basicConfig(
			level=log_level, 
			format='[%(asctime)s] {%(filename)s:%(lineno)d} %(levelname)s - %(message)s',
			handlers=handlers
	)

	stdout_logger = logging.getLogger('STDOUT')
	sl = StreamToLogger(stdout_logger, logging.DEBUG)
	sys.stdout = sl

	stderr_logger = logging.getLogger('STDERR')
	sl = StreamToLogger(stderr_logger, logging.ERROR)
	sys.stderr = sl

	log.info("Initialised logger for new execution")

class StreamToLogger(object):
	"""
	Fake file-like stream object that redirects writes to a logger instance.
	"""
	def __init__(self, logger, log_level=logging.INFO):
		self.logger = logger
		self.log_level = log_level
		self.linebuf = ''

	def write(self, buf):
		for line in buf.rstrip().splitlines():
			 self.logger.log(self.log_level, "STDSTREAM:" + line.rstrip())

	def flush(self):
		pass

def ensure_dir_exists(filename):
	subprocess.call("mkdir -p `dirname " + str(filename) + "`", shell=True)

