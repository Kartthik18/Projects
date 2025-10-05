# config.py
from collections import defaultdict

# ==== Classes & label maps ====
CLASSES = ["pikachu", "charizard", "bulbasaur", "mewtwo"]
label2id = {c: i for i, c in enumerate(CLASSES)}
id2label = {i: c for c, i in label2id.items()}

# ==== Synonyms map (as used in your original code) ====
POKEMON_SYNONYMS = {
    # Pikachu variants
    "electric rat": "pikachu", "yellow mouse": "pikachu", "electric mouse": "pikachu",
    "rodent of sparks": "pikachu", "tiny thunder beast": "pikachu", "yellow electric rat": "pikachu",
    "pika": "pikachu", "ash's partner": "pikachu", "electric type": "pikachu",
    "lightning rodent": "pikachu", "spark-tailed mammal": "pikachu", "volt vermin": "pikachu",
    "electro-rat": "pikachu", "thunder creature": "pikachu", "current-generating mammal": "pikachu",
    "shock-tailed creature": "pikachu", "static-furred mammal": "pikachu", "power-rodent": "pikachu",

    # Charizard variants
    "fire-breathing lizard": "charizard", "orange dragon": "charizard", "fire dragon": "charizard",
    "flame pokemon": "charizard", "flying type": "charizard", "winged inferno": "charizard",
    "flame dragon": "charizard", "scaled fire titan": "charizard", "orange lizard": "charizard",
    "blaze wyrm": "charizard", "inferno drake": "charizard", "fiery serpent": "charizard",
    "combustion lizard": "charizard", "incendiary reptile": "charizard", "pyrokinesis-capable saurian": "charizard",
    "flame-winged beast": "charizard", "thermal draconid": "charizard", "combustion-based saurian": "charizard",

    # Bulbasaur variants
    "seed pokemon": "bulbasaur", "plant toad": "bulbasaur", "grass frog": "bulbasaur",
    "vine pokemon": "bulbasaur", "grass type": "bulbasaur", "sprout toad": "bulbasaur",
    "green seedling": "bulbasaur", "plant reptile": "bulbasaur", "vine beast": "bulbasaur",
    "chlorophyll monster": "bulbasaur", "botanical reptile": "bulbasaur", "leafy quadruped": "bulbasaur",
    "photosynthesis creature": "bulbasaur", "flora-manipulating amphibian": "bulbasaur", "botanical saurian": "bulbasaur",
    "chlorophyll-based vertebrate": "bulbasaur", "photosynthetic saurian": "bulbasaur", "flora-reptile hybrid": "bulbasaur",

    # Mewtwo variants
    "genetic pokemon": "mewtwo", "psychic cat": "mewtwo", "clone pokemon": "mewtwo",
    "legendary": "mewtwo", "psychic type": "mewtwo", "telekinetic predator": "mewtwo",
    "psychic clone": "mewtwo", "genetic experiment": "mewtwo", "synthetic mind weapon": "mewtwo",
    "laboratory horror": "mewtwo", "bio-engineered terror": "mewtwo", "psychic abomination": "mewtwo",
    "clone horror": "mewtwo", "synthetic psychic": "mewtwo", "cerebral construct": "mewtwo",
    "psionically-engineered entity": "mewtwo", "noospheric manifestation": "mewtwo", "consciousness-projected entity": "mewtwo",
}

# ==== Action / Negation words (from your script) ====
ACTION_WORDS = [
    "kill", "eliminate", "destroy", "target", "attack", "neutralize", "get rid of", "erase",
    "take down", "defeat", "wipe out", "terminate", "remove", "execute", "engage",
    "annihilate", "eradicate", "extinguish", "obliterate", "dispatch", "liquidate",
    "exterminate", "decimate"
]

NEGATION_WORDS = [
    "not", "don't", "avoid", "never", "shouldn't", "cannot", "forbidden", "refuse",
    "do not", "don't engage", "ignore", "spare", "protect", "save", "preserve",
    "defend", "safeguard", "shield", "guard"
]

# ==== Tactical filler (trimmed/kept as in your code) ====
TACTICAL_FILLER = [
    "HQ REPORT: Situation analysis regarding unusual activity",
    "Scouts described sightings of", "often accompanied by subtle disruptions in the environment",
    "Additional activity has been noted from", "though they do not appear hostile at present",
    "Field logs: instrumentation drift observed; recalibration recommended",
    "Night-vision readout noisy; expect degraded identification accuracy",
    "Keep monitoring the skies - aerial disturbances are not uncommon in this sector",
    "Communications are patchy; maintain line-of-sight whenever possible",
    "This sector has intermittent interference from legacy comm buoys",
    "Supply convoys have had to reroute due to unstable terrain conditions",
    "HQ analysts believe these disturbances are precursors to a larger conflict",
    "Maintain operational secrecy. HQ will expect a full after-action report.",
    "Use hand signals when verbal comms might reveal position",
    "Draw minimal bloodline; photographic evidence is priority",
    "Re-route patrols to avoid recurring sinkholes along Route 3",
    "Confirm target count by visual confirmation, not by sensor alone",
    "Use thermal masking as a decoy if pursuit is necessary",
    "Use nonlethal methods when the objective allows for capture",
    "Suppression fire authorized only on confirmed hostile contacts",
    "Pre-brief: target identity confirmed via three sensors or witness corroboration",
    "HQ requests photos with scale markers for each contact",
    "Maintain chain-of-custody for discovered artifacts",
    "Long-range sensors indicate sporadic bursts of radiation, possibly linked to latent evolutions",
    "If the target flees, pursue only after authorization from overwatch",
    "Avoid bright illumination near suspected nests - it agitates inhabitants",
    "Multiple witness statements - low confidence - indicate movement at dawn",
    "Calm, low-frequency calls appear to pacify the group temporarily",
    "Radio checkpoint at 02:00 to confirm continued presence",
    "Field units have been reporting strange anomalies in energy readings across the valley",
    "Keep environmental samples for lab analysis (soil, residue, fur)",
    "Intercept logs indicate nonlocal movement patterns at dusk",
    "Tactical note: terrain is uneven; brace for elevation change",
    "Keep a secondary escape corridor clear at all times",
    "Target behavior escalates when food sources are nearby",
    "Extract value intelligence before demolition where possible",
    "Thermal cameras logged irregular heat signatures near the treeline",
    "Local populations call them the 'yellow mouse' in nearby villages",
    "Reports from scouts mention hostile encounters
