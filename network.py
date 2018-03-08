
import _dynet as dy
import random
import numpy as np
import sys


def main(args):
    import dynet as dy
    START, STOP = "<START>", "<STOP>"
    CDIM = args.size_embeddings
    SDIM = args.size_states


    data = [line.strip(" \n\"") for line in open("liste", "r") if line.strip(" \n\"")]
    data = [name.lower() for name in data if "Canton d'" not in name]

    if args.hyphen:
        data = [name for name in data if "-" in name]

    print("Number of training examples: {}".format(len(data)))

    vocabulary = set()

    for city in data:
        vocabulary |= set(city)

    vocabulary.add(START)
    vocabulary.add(STOP)


    vocabulary = sorted(vocabulary)
    voc_code = {c : i for i, c in enumerate(vocabulary)}

    model = dy.Model()

    char_embeddings = model.add_lookup_parameters((len(vocabulary), CDIM))

    lstm = dy.LSTMBuilder(1, CDIM, SDIM, model)

    out_weights = model.add_parameters((len(vocabulary), SDIM))

    trainer = dy.AdamTrainer(model)

    def train_one(name, update=True):
        dy.renew_cg()
        sequence = [START] + list(name) + [STOP]
        coded_seq = [voc_code[c] for c in sequence]

        state = lstm.initial_state()

        loss = dy.zeros(1)
        W = dy.parameter(out_weights)
        for input, target in zip(coded_seq[:-1], coded_seq[1:]):
            embedding = char_embeddings[input]
            state = state.add_input(embedding)

            loss += dy.pickneglogsoftmax(W * state.output(), target)

        if update:
            loss.backward()
            trainer.update()

        return loss.value()

    def recapitalize(name):
        for sep in [" ", "-", "'"]:
            name = sep.join([segment[0].upper() + segment[1:] for segment in name.split(sep) if segment])
        return name

    def generate():
        dy.renew_cg()
        state = lstm.initial_state()
        result = [START]
        result_i = [voc_code[START]]
        W = dy.parameter(out_weights)
        while result[-1] != STOP:
            state = state.add_input(char_embeddings[result_i[-1]])

            output_distrib = dy.softmax(W * state.output()).value()
            output_distrib[voc_code[START]] = 0
            output_distrib /= np.sum(output_distrib)

            prediction = np.random.choice(list(range(len(vocabulary))), p=output_distrib )
            result.append(vocabulary[prediction])
            result_i.append(prediction)


        name = "".join(result[1:-1])
        return recapitalize(name)


    def train(epoch):
        random.shuffle(data)
        sample = data[:1000]
        for i in range(epoch):

            for j, ex in enumerate(data):
                train_one(ex, update=True)
                sys.stderr.write("\r {} % done    ".format(j / len(data) * 100))

            loss = 0
            for j, ex in enumerate(sample):
                loss += train_one(ex, update=False)

            sys.stderr.write("\n# Iteration {}   loss = {}\n".format(i, loss / len(sample)))
            
            predictions = [generate() for i in range(args.num_generation)]
            if args.sncf:
                voie = np.random.choice(list("ABCDEFG"))
                print("Le train, en provenance de {} et à destination de {}, partira voie {}.".format(predictions[0], predictions[1], voie), flush=True)
                print("Il desservira les gares de {} et son terminus {}.".format(", ".join(predictions[2:-1]), predictions[-1]), flush=True)
            else:
                for p in predictions:
                    print(p, flush=True)

            random.shuffle(data)

    train(args.iterations)

if __name__ == "__main__":
    import argparse
    usage = """Train an LSTM to generate French town names"""
    parser = argparse.ArgumentParser(description = usage, formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument("--iterations","-i", type=int, default=50, help="Number of training iterations")
    parser.add_argument("--size-embeddings","-c", type=int, default=50, help="Dimension of character embeddings")
    parser.add_argument("--size-states","-s", type=int, default=50, help="Dimension of LSTM states")
    parser.add_argument("--hyphen", action="store_true", help="Only train on hyphenated town names")
    parser.add_argument("--num-generation", "-n", type=int, default=10, help="Number of generated examples after each iteration")
    parser.add_argument("--sncf", action="store_true")
    args = parser.parse_args()
    if args.sncf:
        args.num_generation = max(args.num_generation, 5)
    main(args)
    
