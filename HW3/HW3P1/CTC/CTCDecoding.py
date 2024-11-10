import numpy as np

class GreedySearchDecoder(object):

    def __init__(self, symbol_set):
        """
        
        Initialize instance variables

        Argument(s)
        -----------

        symbol_set [list[str]]:
            all the symbols (the vocabulary without blank)

        """

        self.symbol_set = symbol_set


    def decode(self, y_probs):
        """

        Perform greedy search decoding

        Input
        -----

        y_probs [np.array, dim=(len(symbols) + 1, seq_length, batch_size)]
            batch size for part 1 will remain 1, but if you plan to use your
            implementation for part 2 you need to incorporate batch_size

        Returns
        -------

        decoded_path [str]:
            compressed symbol sequence i.e. without blanks or repeated symbols

        path_prob [float]:
            forward probability of the greedy path

        """

        decoded_path = []
        blank = 0
        path_prob = 1

        # 1. Iterate over sequence length - len(y_probs[0])
        # 2. Iterate over symbol probabilities
        # 3. update path probability, by multiplying with the current max probability
        # 4. Select most probable symbol and append to decoded_path
        # 5. Compress sequence (Inside or outside the loop)
        seq_length = y_probs.shape[1]
        for t in range(seq_length):
            # Get probabilities for all symbols at time step t
            probs_t = y_probs[:, t, 0]  # Shape: (len(symbols) + 1,)
            k = np.argmax(probs_t)
            max_prob = probs_t[k]

            # Update path probability
            path_prob *= max_prob

            # Append symbol index to decoded path
            decoded_path.append(k)

        # Compress the decoded path: remove blanks and repeated symbols
        compressed_path = []
        previous_symbol = None
        for k in decoded_path:
            if k != blank and k != previous_symbol:
                compressed_path.append(self.symbol_set[k - 1])
                previous_symbol = k
            elif k == blank:
                previous_symbol = None

        decoded_string = ''.join(compressed_path)
        return decoded_string, path_prob


def merge_identical_paths(paths_with_terminal_blank, paths_with_terminal_symbol, blank_path_score, path_score):
    """
    Merge paths that correspond to the same sequence.
    """
    merged_paths = set()
    final_path_score = {}

    # Merge paths that end with a blank
    for path in paths_with_terminal_blank:
        if path in final_path_score:
            final_path_score[path] += blank_path_score[path]
        else:
            final_path_score[path] = blank_path_score[path]
        merged_paths.add(path)

    # Merge paths that end with a symbol
    for path in paths_with_terminal_symbol:
        if path in final_path_score:
            final_path_score[path] += path_score[path]
        else:
            final_path_score[path] = path_score[path]
        merged_paths.add(path)

    return merged_paths, final_path_score


def extend_with_blank(paths_with_terminal_blank, paths_with_terminal_symbol, y, blank_path_score, path_score):
    """
    Extend paths by adding a blank at the current time-step.
    """
    updated_paths_with_terminal_blank = set()
    updated_blank_path_score = {}

    # Extend paths that end with a blank
    for path in paths_with_terminal_blank:
        updated_paths_with_terminal_blank.add(path)
        updated_blank_path_score[path] = blank_path_score[path] * y[0]

    # Extend paths that end with a symbol
    for path in paths_with_terminal_symbol:
        if path in updated_paths_with_terminal_blank:
            updated_blank_path_score[path] += path_score[path] * y[0]
        else:
            updated_paths_with_terminal_blank.add(path)
            updated_blank_path_score[path] = path_score[path] * y[0]
    return updated_paths_with_terminal_blank, updated_blank_path_score


class BeamSearchDecoder(object):

    def __init__(self, symbol_set, beam_width):
        """

        Initialize instance variables

        Argument(s)
        -----------

        symbol_set [list[str]]:
            all the symbols (the vocabulary without blank)

        beam_width [int]:
            beam width for selecting top-k hypotheses for expansion

        """

        self.symbol_set = symbol_set
        self.beam_width = beam_width

    def initialize_paths(self, y):
        """
        Initialize paths with the first time-step probabilities.
        """
        initial_blank_path_score = {}
        initial_path_score = {}

        # Start with a blank path
        path = ''
        initial_blank_path_score[path] = y[0]
        initial_paths_with_final_blank = {path}

        initial_paths_with_final_symbol = set()
        # Initialize paths for each symbol
        for i, symbol in enumerate(self.symbol_set):
            path = symbol
            initial_path_score[path] = y[i + 1]
            initial_paths_with_final_symbol.add(path)
        return (initial_paths_with_final_blank, initial_paths_with_final_symbol,
                initial_blank_path_score, initial_path_score)

    def extend_with_symbol(self, paths_with_terminal_blank, paths_with_terminal_symbol, y, blank_path_score, path_score):
        """
        Extend paths by adding symbols at the current time-step.
        """
        updated_paths_with_terminal_symbol = set()
        updated_path_score = {}

        # Extend paths that end with a blank
        for path in paths_with_terminal_blank:
            for i, symbol in enumerate(self.symbol_set):
                new_path = path + symbol
                updated_paths_with_terminal_symbol.add(new_path)
                updated_path_score[new_path] = blank_path_score[path] * y[i + 1]

        # Extend paths that end with a symbol
        for path in paths_with_terminal_symbol:
            for i, symbol in enumerate(self.symbol_set):
                if symbol == path[-1]:  # Same symbol as the last one
                    new_path = path
                else:
                    new_path = path + symbol

                if new_path in updated_paths_with_terminal_symbol:
                    updated_path_score[new_path] += path_score[path] * y[i + 1]
                else:
                    updated_paths_with_terminal_symbol.add(new_path)
                    updated_path_score[new_path] = path_score[path] * y[i + 1]
        return updated_paths_with_terminal_symbol, updated_path_score

    def prune(self, paths_with_terminal_blank, paths_with_terminal_symbol, blank_path_score, path_score):
        """
        Prune the paths to keep only the top `beam_width` paths.
        """
        pruned_blank_path_score = {}
        pruned_path_score = {}
        pruned_paths_with_terminal_blank = set()
        pruned_paths_with_terminal_symbol = set()

        # Collect all scores
        score_list = [blank_path_score[p] for p in paths_with_terminal_blank] + \
                     [path_score[p] for p in paths_with_terminal_symbol]
        score_list.sort(reverse=True)
        cutoff = score_list[self.beam_width - 1] if len(score_list) >= self.beam_width else score_list[-1]

        # Prune paths with blank
        for path in paths_with_terminal_blank:
            if blank_path_score[path] >= cutoff:
                pruned_paths_with_terminal_blank.add(path)
                pruned_blank_path_score[path] = blank_path_score[path]

        # Prune paths with symbols
        for path in paths_with_terminal_symbol:
            if path_score[path] >= cutoff:
                pruned_paths_with_terminal_symbol.add(path)
                pruned_path_score[path] = path_score[path]

        return (pruned_paths_with_terminal_blank, pruned_paths_with_terminal_symbol,
                pruned_blank_path_score, pruned_path_score)

    def decode(self, y_probs):
        """
        Perform beam search decoding

        Input
        -----
        y_probs [np.array, dim=(len(symbols) + 1, seq_length, batch_size)]
            Batch size for part 1 will remain 1, but for part 2, incorporate batch_size.

        Returns
        -------
        forward_path [str]:
            The symbol sequence with the best path score (forward probability)
        merged_path_scores [dict]:
            All the final merged paths with their scores
        """
        num_symbols, seq_len, batch_size = y_probs.shape
        y_probs = y_probs[:, :, 0]  # Shape: (num_symbols, seq_len) for batch size of 1

        # Initialize paths
        (new_paths_with_terminal_blank, new_paths_with_terminal_symbol,
         new_blank_path_score, new_path_score) = self.initialize_paths(y_probs[:, 0])

        for t in range(1, seq_len):
            # Prune paths
            (paths_with_terminal_blank, paths_with_terminal_symbol,
             blank_path_score, path_score) = self.prune(new_paths_with_terminal_blank,
                                                        new_paths_with_terminal_symbol,
                                                        new_blank_path_score,
                                                        new_path_score)
            # Extend paths with a blank
            (new_paths_with_terminal_blank, new_blank_path_score) = extend_with_blank(
                paths_with_terminal_blank, paths_with_terminal_symbol, y_probs[:, t], blank_path_score, path_score)

            # Extend paths with symbols
            (new_paths_with_terminal_symbol, new_path_score) = self.extend_with_symbol(
                paths_with_terminal_blank, paths_with_terminal_symbol, y_probs[:, t], blank_path_score, path_score)

        # Merge identical paths
        merged_paths, final_path_score = merge_identical_paths(
            new_paths_with_terminal_blank, new_paths_with_terminal_symbol,
            new_blank_path_score, new_path_score)

        # Determine the best path
        best_path = max(final_path_score, key=final_path_score.get)
        return best_path, final_path_score