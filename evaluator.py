import sys
import models
import math

### Evaluator
# This class adapts the code from the grading function to compute
# all phrase-to-phrase alignments, and assign a score to a 
# translated sentence. It is used to compare two translations.
# It can also be used as a scoring function by the greedy decoder.
class Evaluator:
	def __init__(self, opts):
		self.opts = opts
		self.tm = models.TM(opts.tm, sys.maxint)
		self.lm = models.LM(opts.lm)
		self.french = [tuple(line.strip().split()) for line in open(opts.input).readlines()]
		# tm should translate unknown words as-is with probability 1
		for word in set(sum(self.french,())):
		  if (word,) not in self.tm:
		    self.tm[(word,)] = [models.phrase(word, 0.0)]

	# returns the total log probability for a translation,
	# using the grader's algorithm.
	def grade_score(self, f, e):
		alignments = self.get_alignments(f, e)
		score = self.grade_with_alignments(f, e, alignments)
		return score

	# Computes all possible phrase-to-phrase alignments.
	def get_alignments(self, f, e):
		alignments = [[] for _ in e]
		for fi in xrange(len(f)):
		  for fj in xrange(fi+1,len(f)+1):
		    if f[fi:fj] in self.tm:
		      for phrase in self.tm[f[fi:fj]]:
		        ephrase = tuple(phrase.english.split())
		        for ei in xrange(len(e)+1-len(ephrase)):
		          ej = ei+len(ephrase)
		          if ephrase == e[ei:ej]:
		            alignments[ei].append((ej, phrase.logprob, fi, fj))
		return alignments

	# Grade a sentence with a given set of alignments. Lets you 
	# grade numerous sentences without the overhead of computing alignments
	# each time.
	def grade_with_alignments(self, f, e, alignments):
		lm_state = self.lm.begin()
		total_logprob = 0.0
		lm_logprob = 0.0
		for word in e + ("</s>",):
		  (lm_state, word_logprob) = self.lm.score(lm_state, word)
		  lm_logprob += word_logprob
		total_logprob += lm_logprob
		# Compute sum of probability of all possible alignments by dynamic programming.
		chart = [{} for _ in e] + [{}]
		chart[0][0] = 0.0
		for ei, sums in enumerate(chart[:-1]):
		  for v in sums:
		    for ej, logprob, fi, fj in alignments[ei]:
		      if self.bitmap(range(fi,fj)) & v == 0:
		        new_v = self.bitmap(range(fi,fj)) | v
		        if new_v in chart[ej]:
		          chart[ej][new_v] = self.logadd10(chart[ej][new_v], sums[v]+logprob)
		        else:
		          chart[ej][new_v] = sums[v]+logprob
		goal = self.bitmap(range(len(f)))
		if goal in chart[len(e)]:
		  total_logprob += chart[len(e)][goal]
		else:
		  total_logprob = -10000.0
		return total_logprob

	def logadd10(self, x,y):
	  """ Addition in logspace (base 10): if x=log(a) and y=log(b), returns log(a+b) """
	  return x + math.log10(1 + pow(10,y-x))

	def bitmap(self, sequence):
		""" Generate a coverage bitmap for a sequence of indexes """
		return reduce(lambda x,y: x|y, map(lambda i: long('1'+'0'*i,2), sequence), 0)