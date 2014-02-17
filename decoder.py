### Decoder - Statistical Machine Translation
### Sean Welleck | 2014
#
# A module for decoding setences of a source language into a target language,
# given a language model and translation model.
#
# Usage: target_sentence = decoder.decode(source_sentence, lm, tm)

from collections import namedtuple
import copy

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

# Performs decoding with a language model and translation model.
# Contains multiple decoding strategies, which are combined in decode().
class Decoder:
	def __init__(self, lm, tm, opts):
		self.lm = lm
		self.tm = tm
		self.opts = opts

	# Decode a source sentence string into a target sentence string.
	def decode(self, source):
		seed = self.monotone_decode(source)
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
		iters = 100
		current = seed
		for i in xrange(iters):
		  s_current = self.score(current, source)
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
	# E.g. ('si', 'aucun', 'autre') -> [('si', 'aucun'), ('autre')],
	#																	 [('si'), ('aucun', 'autre')]
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

	# TODO
	# Decode a sentence using a non-monotone stack decoder.
	# Input: 	source - a sentence string in the source language
	# Output: winner - the highest scoring Hypothesis
	def stack_decode(self, source):
		return self.Hypothesis("", "", 0.00)
