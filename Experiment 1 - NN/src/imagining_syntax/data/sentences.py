import random
from imagining_syntax.data.distributions import create_distribution

verbs_singular = ["challenges", "collapses", "twirls", "orates", "sings", "paints", "listens", "embezzles", "crafts", "realizes", "lassos", "charms", "whistles", "dreams", "solves", "arches", "incites", "bridges", "denotes", "chances", "builds", "teaches", "writes", "runs", "drives", "flies", "swims", "climbs", "reads", "bakes", "dances", "jumps", "hunts", "fishes", "mines", "farms", "trades", "protects", "navigates", "leads"]
verbs_plural = ["challenge", "collapse", "twirl", "orate", "sing", "paint", "listen", "embezzle", "craft", "realize", "lasso", "charm", "whistle", "dream", "solve", "arch", "incite", "bridge", "denote", "chance", "build", "teach", "write", "run", "drive", "fly", "swim", "climb", "read", "bake", "dance", "jump", "hunt", "fish", "mine", "farm", "trade", "protect", "navigate", "lead"]

nouns_singular = ["challenger", "collapser", "twirler", "orator", "singer", "painter", "listener", "embezzler", "crafter", "realizer", "lassoer", "charmer", "whistler", "dreamer", "solver", "archer", "inciter", "bridger", "denoter", "chancer", "builder", "teacher", "writer", "runner", "driver", "flyer", "swimmer", "climber", "reader", "baker", "dancer", "jumper", "hunter", "fisher", "miner", "farmer", "trader", "protector", "navigator", "leader"]
nouns_plural = ["challengers", "collapsers", "twirlers", "orators", "singers", "painters", "listeners", "embezzlers", "crafters", "realizers", "lassoers", "charmers", "whistlers", "dreamers", "solvers", "archers", "inciters", "bridgers", "denoters", "chancers", "builders", "teachers", "writers", "runners", "drivers", "flyers", "swimmers", "climbers", "readers", "bakers", "dancers", "jumpers", "hunters", "fishers", "miners", "farmers", "traders", "protectors", "navigators", "leaders"]

determiner = ["the"]

prepositions = ["by", "near"]

N = len(verbs_singular)

def generate_sentence(param_value, distribution_type="zipfian", prep_obj_mismatch=False, both_pps_present=False, noun_OOD=False, vocab_size=None, unseen_count=None):
    # Set defaults if not provided
    if vocab_size is None:
        vocab_size = N
    if unseen_count is None:
        unseen_count = 10
    
    sentence_number = random.choice(["singular", "plural"])
    
    if sentence_number == "plural":
        verb_list = verbs_plural[:vocab_size]
        noun_list = nouns_plural[:vocab_size]
    else:
        verb_list = verbs_singular[:vocab_size]
        noun_list = nouns_singular[:vocab_size]

    verb_index = random.choice(list(range(vocab_size)))
    verb = verb_list[verb_index]
    # put verb index function here!
    # use create distribution function here which takes in C, and len(verbs_singular) as n
    # Make another function that takes in 2 args(prob_distribution_list, verb_index)
    # Should return that list starting at verb_index with everything else
    # Wrapped around to the front
    # list[verb_index:] 
    # rotate the probability index so that it starts the correct verb index

    if noun_OOD:
        distribution = [0 for _ in range(vocab_size-unseen_count)] + [1]*unseen_count
        preposition_distribution = [0 for _ in range(vocab_size-unseen_count)] + [1]*unseen_count
    else:
        distribution = create_distribution(distribution_type, param_value, vocab_size, unseen_count)
        preposition_distribution = [1 for _ in range(vocab_size-unseen_count)] + [0]*unseen_count
        
    noun_index = (random.choices(list(range(vocab_size)), weights=distribution)[0] + verb_index) % vocab_size
    #print(noun_index)
    #print(verb_index)

    subject_noun = noun_list[noun_index]

    subject = ["the", subject_noun]  

    if random.choice([True, False]) or both_pps_present == True:

        pobj_index = (random.choices(list(range(vocab_size)), weights=preposition_distribution)[0] + verb_index) % vocab_size 
        
        if sentence_number == "plural" and prep_obj_mismatch == False:
            prep_object = nouns_plural[pobj_index]

        if sentence_number == "plural" and prep_obj_mismatch == True:
            prep_object = nouns_singular[pobj_index]

        if sentence_number == "singular" and prep_obj_mismatch == False:
            prep_object = nouns_singular[pobj_index]

        if sentence_number == "singular"and prep_obj_mismatch == True:
            prep_object = nouns_plural[pobj_index]

        preposition = random.choice(prepositions)
        subject = subject + [preposition, "the", prep_object]

    if random.choice([True, False]) or both_pps_present == True:
        pobj_index = (random.choices(list(range(vocab_size)), weights=preposition_distribution)[0] + verb_index) % vocab_size 
        
        if sentence_number == "plural" and prep_obj_mismatch == False:
            prep_object = nouns_plural[pobj_index]

        if sentence_number == "plural" and prep_obj_mismatch == True:
            prep_object = nouns_singular[pobj_index]

        if sentence_number == "singular" and prep_obj_mismatch == False:
            prep_object = nouns_singular[pobj_index]

        if sentence_number == "singular"and prep_obj_mismatch == True:
            prep_object = nouns_plural[pobj_index]
        
        preposition = random.choice(prepositions)
        subject = [preposition, "the", prep_object] + subject


    sentence = subject + [verb]

    return " ".join(sentence)


#for _ in range(30):
    #print(generate_sentence(1.0))
#print(sum(create_distribution(0.5)))
#print(create_distribution(0.5))



