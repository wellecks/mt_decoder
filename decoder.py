### Decoder - Statistical Machine Translation
### Sean Welleck | 2014
#
# A module for decoding sentences of a source language into a target language,
# given a language model and translation model.
#
# Usage: target_sentence = decoder.decode(source_sentence, lm, tm)

from collections import namedtuple
import copy
import evaluator
import sys

# The top-level function for the decoding algorithm. Decodes a 
# source sentence, given a language model and a translation model. 
# Input:	source 	- a sentence string in the source language, e.g. french
#					lm     	- a trained language model
#					tm			- a trained translation model
#					opts		- command line options
# Output:	target  - a sentence string in the target language, e.g. english
def decode(source, lm, tm, opts):
	d = Decoder(lm, tm, opts)
	return d.decode(source)

def combine(dfile1, dfile2, lm, tm, opts):
	d = Decoder(lm, tm, opts)
	d.combine_decodings(dfile1, dfile2)

# Performs decoding with a language model and translation model.
# Contains multiple decoding strategies, which are combined in decode().
class Decoder:
	def __init__(self, lm, tm, opts):
		self.lm = lm
		self.tm = tm
		self.opts = opts

	# Decode a source sentence string into a target sentence string.
	def decode(self, source):
		#seed = self.monotone_decode(source)
		seed = self.stack_decode(source)
		decoded = self.greedy_decode(source, seed)
		return self.print_phrases(decoded)

	# Decode a sentence using a monotone decoder.
	# Input: 	source 	- a sentence string in the source language
	# Output: phrases - the highest scoring hypothesis as formatted 
	#										by hyp_to_phrases().
	def monotone_decode(self, source):
		hypothesis = namedtuple("hypothesis", "logprob, lm_state, predecessor, phrase, fphrase")
		initial_hypothesis = hypothesis(0.0, self.lm.begin(), None, None, None)
		# create a stack for each number-of-words-translated
		stacks = [{} for _ in source] + [{}]
		# in the zero'th stack, map start symbol to empty hypothesis
		stacks[0][self.lm.begin()] = initial_hypothesis
		# for each stack
		for i, stack in enumerate(stacks[:-1]):
		  # histogram pruning (just chop off n - s worst scoring hypotheses)
		  for h in sorted(stack.itervalues(),key=lambda h: -h.logprob)[:self.opts.s]: # prune
		    # look to the next consecutive words
		    for j in xrange(i+1,len(source)+1):
		      # if the sequence is in the translation model
		      if source[i:j] in self.tm:
		        # [(english, logprob)]
		        for phrase in self.tm[source[i:j]]:
		          logprob = h.logprob + phrase.logprob
		          lm_state = h.lm_state
		          # calculate p(e) using the language model
		          for word in phrase.english.split():
		            (lm_state, word_logprob) = self.lm.score(lm_state, word)
		            logprob += word_logprob
		          logprob += self.lm.end(lm_state) if j == len(source) else 0.0
		          new_hypothesis = hypothesis(logprob, lm_state, h, phrase, source[i:j])
		          # lm_state contains the sequence of words (e.g. ('<s>', 'honourable'))
		          if lm_state not in stacks[j] or stacks[j][lm_state].logprob < logprob: # second case is recombination
		            stacks[j][lm_state] = new_hypothesis 
		winner = max(stacks[-1].itervalues(), key=lambda h: h.logprob)
		return self.hyp_to_phrases(winner)

	# Converts a hypothesis into a list of phrases.
  # Input:  hyp - hypothesis
  # Output: ps  - [(phrase, french)]
  #               e.g. [(phrase(english='honourable', logprob=0.0), ('honorables',))]
	def hyp_to_phrases(self, hyp):
		phrases = []
		def get_phrases(hyp, ps):
			if hyp == None:
				return
			else:
				ps.insert(0, (hyp.phrase, hyp.fphrase))
				get_phrases(hyp.predecessor, ps)
		get_phrases(hyp, phrases)
		return phrases

	# Decodes a sentence with a greedy, hill-climbing decoder. 
	# Requires a 'seed' sentence to start with.
	# Input: 	source - a sentence string in the source language
	# 				seed 	 - an initial hypothesis
	# Output: the highest scoring hypothesis as formatted by hyp_to_phrases.
	def greedy_decode(self, source, seed):
		ev = evaluator.Evaluator(self.opts)
		e = tuple([ep.english for (ep, _) in seed if ep != None])
		alignments = ev.get_alignments(source, e)
		iters = 100
		current = seed
		for i in xrange(iters):
		  #s_current = self.score(current, source)
		  s_current = self.score_with_grader(source, e, alignments, ev)
		  s = s_current
		  for h in self.neighborhood(current):
		    c = self.score(h, source)
		    if c > s:
		      s = c
		      best = h
		  if s == s_current:
		    return current
		  else:
		    current = best
		return current

	# Scores a translation using the language and translation models.
	# Used by the greedy decoder.
	def score(self, ps, source):
	  # language model score
	  lm_prob = 0.0
	  lm_state = self.lm.begin()
	  num_f_translated = 0
	  for n, (ep, fp) in enumerate(ps):
	    if ep != None and fp != None:
	      num_f_translated += len(fp)
	      for word in ep.english.split():
	        (lm_state, word_logprob) = self.lm.score(lm_state, word)
	        lm_prob += word_logprob
	      lm_prob += self.lm.end(lm_state) if num_f_translated == len(source) else 0.0

	  # translation model score
	  tm_prob = 0.0
	  for (ep, fp) in ps:
	    if ep != None:
	      tm_prob += ep.logprob

	  return (lm_prob + tm_prob)

	def score_with_grader(self, f, e, alignments, ev):
		score = ev.grade_with_alignments(f, e, alignments)
		return score

	# The possible next-steps for the hill climbing algorithm.
	def neighborhood(self, ps):
	  return self.swap(ps) + self.merge(ps) + self.replace(ps) + self.split(ps)

	# Swap each unique pair of phrases, and return as a list of phrases.
	def swap(self, ps):
	  swaps = []
	  for i in xrange(len(ps)-1):
	    for j in xrange(i, len(ps)):
	      swapped = copy.deepcopy(ps)
	      temp = swapped[i]
	      swapped[i] = swapped[j]
	      swapped[j] = temp
	      swaps.append(swapped)
	  return swaps

	# For all phrases in the input list,
	# replaces a single phrase with each of its alternative definitions.
	# Return all of the new phrases in a list.
	def replace(self, ps):
	  replaces = []
	  for n, p in enumerate(ps):
	    if p[1] in self.tm:
	      ts = self.tm[p[1]]
	      for t in ts:
	        if p[0] != t:
	          replaced = copy.deepcopy(ps)
	          replaced[n] = (t, p[1])
	          replaces.append(replaced)
	  return replaces

	# Merge consecutive source phrases into a single phrase, if
	# the merged translation exists. Currently does 2- and 3-phrase
	# merges.
	def merge(self, ps):
	  merges = []
	  for i in xrange(1, len(ps)-1):
	    f1 = ps[i][1]
	    f2 = ps[i+1][1]
	    if f1 and f2 and (f1 + f2) in self.tm:
	      for t in self.tm[f1+f2]:
	        merged = copy.deepcopy(ps)
	        merged.remove(ps[i+1])
	        merged[i] = (t, f1+f2)
	        merges.append(merged)
	  if len(ps) >= 3:
	    for i in xrange(1, len(ps)-2):
	      f1 = ps[i][1]
	      f2 = ps[i+1][1]
	      f3 = ps[i+2][1]
	      if f1 and f2 and f3 and (f1 + f2 + f3) in self.tm:
	        for t in self.tm[f1+f2+f3]:
	          merged = copy.deepcopy(ps)
	          merged.remove(ps[i+1])
	          merged.remove(ps[i+2])
	          merged[i] = (t, f1+f2+f3)
	          merges.append(merged)
	  return merges

	# Splits each multiple-word source phrase into
	# two source phrases, if their translation exist. 
	# Tries all possible 2-splits.
	# E.g. ('si', 'aucun', 'autre') -> [('si', 'aucun'), ('autre')], [('si'), ('aucun', 'autre')]
	def split(self, ps):
	  splits = []
	  for n, i in enumerate(ps):
	    french_phrase = ps[n][1]
	    if french_phrase != None:
	      if len(french_phrase) > 1:
	        for j in xrange(1, len(french_phrase)):
	          s1 = french_phrase[0:j]
	          s2 = french_phrase[j:]
	          if s1 in self.tm and s2 in self.tm:
	            for ts1 in self.tm[s1]:
	              for ts2 in self.tm[s2]:
	                spl = copy.deepcopy(ps)
	                spl[n] = (ts1, s1)
	                spl.insert(n+1, (ts2, s2))
	                splits.append(spl)
	  return splits

	# Returns the english portion of a list of phrases as 
	# space-separated string.
	def print_phrases(self, phrases):
		s = ""
		for p in phrases:
		  if p[0] != None:
		    s += p[0].english + " "
		return s

	# Decode a sentence using a non-monotone stack decoder.
	# Input: source - a sentence string in the source language
	# Output: winner - the highest scoring Hypothesis
	def stack_decode(self, source):
		hypo = namedtuple("hypo", "logprob, lm_state, predecessor, phrase, marked, end_i, fphrase")
		# 1 if we've translated the word at the index
		marked = [0 for _ in source] 
		initial_hypothesis = hypo(0.0, self.lm.begin(), None, None, marked, 0, None)

		# create a stack for each number-of-words-translated
		stacks = [{} for _ in source] + [{}]
		# in the zero'th stack, map start symbol to empty hypothesis
		stacks[0][self.lm.begin()] = initial_hypothesis
		for i, stack in enumerate(stacks[:-1]):
		  for hyp in sorted(stack.itervalues(), key=lambda h: -h.logprob)[:self.opts.s]:
		    # get the translation options for this hypothesis
		    options = self.get_trans_options(hyp, source)

		    # for each translation option
		    for (phrase, idxs) in options:
		      start_ind = idxs[0]
		      end_ind = idxs[1]
		      # add the log probability from the translation model
		      logprob = hyp.logprob + phrase.logprob
		      lm_state = hyp.lm_state

		      # evaluate the english phrase using the language model
		      for word in phrase.english.split():
		        (lm_state, word_logprob) = self.lm.score(lm_state, word)
		        logprob += word_logprob
		        logprob += self.lm.end(lm_state) if end_ind == len(source)-1 else 0.0
		      marked = copy.deepcopy(hyp.marked)
		      # mark the word sequence that we're translating to denote
		      # that the words have been translated in this hypothesis
		      for x in xrange(start_ind, end_ind):
		        marked[x] = 1
		      num_marked = len(filter(lambda x: x == 1, marked))
		      tmark = tuple(marked)
		      # create a new hypothesis
		      new_hypothesis = hypo(logprob, lm_state, hyp, phrase, marked, end_ind, source[start_ind:end_ind])
		      if tmark not in stacks[num_marked] or stacks[num_marked][tmark].logprob < logprob: # second case is recombination
		        stacks[num_marked][tmark] = new_hypothesis
		winner = max(stacks[-1].itervalues(), key=lambda h: h.logprob)
		return self.hyp_to_phrases(winner)

	# given a hypothesis, get all remaining translation options
	# the options should be drawn from words that have yet to be translated
	# sequences must only be contiguous words of the foreign sentence
	def get_trans_options(self, h, f):
	  options = []
	  for fi in xrange(len(f)):
	    for fj in xrange(fi+1, len(f)+1):
	      # check if the range is unmarked
	      unmarked = all(lambda x: h.marked[x]==0 for m in range(fi, fj))
	      if unmarked:
	        if f[fi:fj] in self.tm:
	          phrases = self.tm[f[fi:fj]]
	          for p in phrases:
	            options.append((p, (fi, fj)))
	  return options

	# Combines two files containing decodings by choosing, for each sentence,
	# the higher scoring decoding. Writes to a file.
	# Input:	dfile1 - a file with sentence decodings
	# 				dfile2 - a file with sentence decodings
	# Output:	prints out combined decodings.
	def combine_decodings(self, dfile1, dfile2):
		e = evaluator.Evaluator(self.opts)
		decodings1 = [tuple(line.strip().split()) for line in open(dfile1).readlines()]
		decodings2 = [tuple(line.strip().split()) for line in open(dfile2).readlines()]
		french = [tuple(line.strip().split()) for line in open(self.opts.input).readlines()]
		for n, (f, (d1, d2)) in enumerate(zip(french, zip(decodings1, decodings2))):
			score1 = e.grade_score(f, d1)
			score2 = e.grade_score(f, d2)
			if score1 > score2:
				print(" ".join(d1))
			else:
				print(" ".join(d2))
			sys.stderr.write("Combined %d sentences." % n)

