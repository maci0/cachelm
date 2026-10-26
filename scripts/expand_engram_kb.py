#!/usr/bin/env python3
"""Expand host Engram-lite KB (ll3m/kb_facts.json) without touching weights.

Usage:
  uv run python scripts/expand_engram_kb.py
  uv run python scripts/expand_engram_kb.py --dry-run

Rules:
  - Never overwrite an existing flat key with a different value.
  - Norm keys match ll3m.tools._norm_key (strip + lower + collapse space).
  - Recompute flat, n_entries, n_keys from entries.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from datetime import date
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
KB_PATH = ROOT / "ll3m" / "kb_facts.json"


def norm_key(s: str) -> str:
    return re.sub(r"\s+", " ", s.strip().lower())


# New canonical entries: (id, value, aliases)
# Prefer school/agent-useful facts; keep values short for tool results.
NEW_ENTRIES: list[tuple[str, str, list[str]]] = [
    # Phase / units
    ("boiling point of water", "100 degrees Celsius at 1 atm", [
        "boiling point water", "water boiling point", "boil water temp",
    ]),
    ("freezing point of water", "0 degrees Celsius at 1 atm", [
        "freezing point water", "water freezing point", "ice melting point",
        "melting point of ice",
    ]),
    ("fahrenheit water freeze", "32 degrees Fahrenheit", [
        "water freezes fahrenheit", "freezing point water f",
    ]),
    ("fahrenheit water boil", "212 degrees Fahrenheit", [
        "water boils fahrenheit", "boiling point water f",
    ]),
    ("c to f formula", "F = C * 9/5 + 32", ["celsius to fahrenheit", "convert c to f"]),
    ("f to c formula", "C = (F - 32) * 5/9", ["fahrenheit to celsius", "convert f to c"]),
    ("si unit length", "meter", ["si length", "unit of length si"]),
    ("si unit mass", "kilogram", ["si mass", "unit of mass si"]),
    ("si unit time", "second", ["si time", "unit of time si"]),
    ("si unit current", "ampere", ["si current", "unit of current si"]),
    ("si unit temperature", "kelvin", ["si temperature", "unit of temperature si"]),
    ("si unit amount", "mole", ["si amount", "unit of amount si"]),
    ("si unit luminous", "candela", ["si luminous intensity"]),
    ("atmospheric pressure", "101325 pascals or 1 atm", [
        "standard pressure", "1 atm in pascals", "sea level pressure",
    ]),
    ("sound speed air", "about 343 meters per second at 20 C", [
        "speed of sound air", "speed of sound",
    ]),
    ("human body temperature", "about 37 degrees Celsius", [
        "normal body temperature", "body temperature c",
    ]),
    ("blood ph", "about 7.4", ["ph of blood", "human blood ph"]),
    ("ph of pure water", "7", ["water ph", "neutral ph"]),
    ("acid ph", "less than 7", ["acids ph"]),
    ("base ph", "greater than 7", ["bases ph", "alkaline ph"]),
    # Biology
    ("dna full name", "deoxyribonucleic acid", [
        "dna full form", "dna stands for", "what does dna stand for",
    ]),
    ("rna full name", "ribonucleic acid", [
        "rna full form", "rna stands for", "what does rna stand for",
    ]),
    ("atp full name", "adenosine triphosphate", ["atp full form", "atp stands for"]),
    ("chromosomes human", "46", ["human chromosomes", "number of human chromosomes"]),
    ("amino acids count", "20 common", ["how many amino acids", "number of amino acids"]),
    ("photosynthesis formula", "6CO2 + 6H2O + light -> C6H12O6 + 6O2", [
        "photosynthesis equation", "equation for photosynthesis",
    ]),
    ("cellular respiration formula", "C6H12O6 + 6O2 -> 6CO2 + 6H2O + ATP", [
        "respiration equation", "cellular respiration equation",
    ]),
    # Constants / physics
    ("avogadro number", "6.022e23 per mole", [
        "avogadro constant", "avogadros number", "n_a",
    ]),
    ("planck constant", "6.626e-34 joule seconds", ["plancks constant", "h planck"]),
    ("boltzmann constant", "1.381e-23 joule per kelvin", ["k_b", "boltzmanns constant"]),
    ("gas constant r", "8.314 joule per mole kelvin", [
        "ideal gas constant", "universal gas constant",
    ]),
    ("electron charge", "about 1.602e-19 coulomb", [
        "elementary charge", "charge of electron",
    ]),
    ("electron mass", "about 9.109e-31 kilogram", ["mass of electron"]),
    ("proton mass", "about 1.673e-27 kilogram", ["mass of proton"]),
    ("neutron mass", "about 1.675e-27 kilogram", ["mass of neutron"]),
    ("gravity constant g", "about 9.80665 m/s^2", [
        "standard gravity", "g acceleration", "earth surface g",
    ]),
    ("speed of light vacuum", "299792458 meters per second", [
        "c light vacuum", "speed of light in vacuum",
    ]),
    # Space / earth
    ("earth circumference", "about 40075 kilometers", [
        "circumference of earth", "earth equatorial circumference",
    ]),
    ("earth radius", "about 6371 kilometers", ["radius of earth", "mean earth radius"]),
    ("moon distance", "about 384400 kilometers", [
        "distance to moon", "earth moon distance",
    ]),
    ("sun distance", "about 149.6 million kilometers", [
        "distance to sun", "earth sun distance", "1 au",
    ]),
    ("light year", "about 9.46 trillion kilometers", [
        "length of light year", "one light year",
    ]),
    ("largest planet", "Jupiter", ["biggest planet", "largest planet solar system"]),
    ("smallest planet", "Mercury", ["smallest planet solar system"]),
    ("hottest planet", "Venus", ["hottest planet solar system"]),
    ("closest star", "Proxima Centauri", ["nearest star", "closest star to sun"]),
    ("milky way type", "barred spiral galaxy", ["milky way galaxy type"]),
    ("earth age", "about 4.54 billion years", ["age of earth", "how old is earth"]),
    ("universe age", "about 13.8 billion years", ["age of universe", "how old is universe"]),
    ("periodic table elements", "118", [
        "how many elements", "number of elements", "elements on periodic table",
    ]),
    # Math / CS
    ("area circle", "pi * r squared", ["circle area formula", "area of a circle"]),
    ("circumference circle", "2 * pi * r", [
        "circle circumference formula", "circumference of a circle",
    ]),
    ("pythagorean theorem", "a squared + b squared = c squared", [
        "pythagoras theorem", "pythagorean equation",
    ]),
    ("e approximate", "2.71828", ["eulers number", "base of natural log"]),
    ("golden ratio", "about 1.618", ["phi golden ratio", "golden mean"]),
    ("square root of 2", "about 1.414", ["sqrt 2", "root 2"]),
    ("square root of 3", "about 1.732", ["sqrt 3", "root 3"]),
    ("number of bits in byte", "8", ["bits in a byte", "how many bits in a byte"]),
    ("http default port", "80", ["port for http"]),
    ("https default port", "443", ["port for https"]),
    ("ssh default port", "22", ["port for ssh"]),
    ("binary base", "2", ["base of binary"]),
    ("hexadecimal base", "16", ["base of hex", "hex base"]),
    ("decimal base", "10", ["base of decimal"]),
    # Counts / calendar
    ("continents count", "7", ["how many continents", "number of continents"]),
    ("oceans count", "5", ["how many oceans", "number of oceans"]),
    ("months in year", "12", ["how many months", "number of months"]),
    ("hours in day", "24", ["how many hours in a day"]),
    ("minutes in hour", "60", ["how many minutes in an hour"]),
    ("seconds in minute", "60", ["how many seconds in a minute"]),
    ("days in year", "365 (366 leap)", ["how many days in a year"]),
    ("weeks in year", "52", ["how many weeks in a year"]),
    # Geo / capitals (extras)
    ("capital of poland", "Warsaw", ["poland capital", "capital poland"]),
    ("capital of sweden", "Stockholm", ["sweden capital", "capital sweden"]),
    ("capital of norway", "Oslo", ["norway capital", "capital norway"]),
    ("capital of finland", "Helsinki", ["finland capital", "capital finland"]),
    ("capital of denmark", "Copenhagen", ["denmark capital", "capital denmark"]),
    ("capital of netherlands", "Amsterdam", [
        "netherlands capital", "capital netherlands", "holland capital",
    ]),
    ("capital of belgium", "Brussels", ["belgium capital", "capital belgium"]),
    ("capital of switzerland", "Bern", ["switzerland capital", "capital switzerland"]),
    ("capital of austria", "Vienna", ["austria capital", "capital austria"]),
    ("capital of portugal", "Lisbon", ["portugal capital", "capital portugal"]),
    ("capital of greece", "Athens", ["greece capital", "capital greece"]),
    ("capital of turkey", "Ankara", ["turkey capital", "capital turkey"]),
    ("capital of south korea", "Seoul", [
        "south korea capital", "capital south korea", "rok capital",
    ]),
    ("capital of north korea", "Pyongyang", [
        "north korea capital", "capital north korea", "dprk capital",
    ]),
    ("capital of mexico", "Mexico City", ["mexico capital", "capital mexico"]),
    ("capital of argentina", "Buenos Aires", [
        "argentina capital", "capital argentina",
    ]),
    ("capital of chile", "Santiago", ["chile capital", "capital chile"]),
    ("capital of nigeria", "Abuja", ["nigeria capital", "capital nigeria"]),
    ("capital of kenya", "Nairobi", ["kenya capital", "capital kenya"]),
    ("capital of indonesia", "Jakarta", ["indonesia capital", "capital indonesia"]),
    ("capital of thailand", "Bangkok", ["thailand capital", "capital thailand"]),
    ("capital of vietnam", "Hanoi", ["vietnam capital", "capital vietnam"]),
    ("capital of philippines", "Manila", [
        "philippines capital", "capital philippines",
    ]),
    ("capital of new zealand", "Wellington", [
        "new zealand capital", "capital new zealand",
    ]),
    ("capital of ireland", "Dublin", ["ireland capital", "capital ireland"]),
    ("capital of scotland", "Edinburgh", ["scotland capital", "capital scotland"]),
    ("capital of wales", "Cardiff", ["wales capital", "capital wales"]),
    ("capital of ukraine", "Kyiv", ["ukraine capital", "capital ukraine", "kiev"]),
    ("capital of saudi arabia", "Riyadh", [
        "saudi arabia capital", "capital saudi arabia",
    ]),
    ("capital of iran", "Tehran", ["iran capital", "capital iran"]),
    ("capital of iraq", "Baghdad", ["iraq capital", "capital iraq"]),
    ("capital of pakistan", "Islamabad", ["pakistan capital", "capital pakistan"]),
    ("capital of bangladesh", "Dhaka", ["bangladesh capital", "capital bangladesh"]),
    ("capital of malaysia", "Kuala Lumpur", [
        "malaysia capital", "capital malaysia",
    ]),
    ("capital of singapore", "Singapore", ["singapore capital"]),
    ("largest ocean", "Pacific Ocean", ["biggest ocean", "largest ocean on earth"]),
    ("tallest mountain", "Mount Everest", [
        "highest mountain", "tallest mountain on earth", "everest height rank",
    ]),
    ("deepest ocean trench", "Mariana Trench", [
        "deepest trench", "deepest part of ocean",
    ]),
    ("largest continent", "Asia", ["biggest continent"]),
    ("smallest continent", "Australia", ["smallest continent on earth"]),
    ("hardest natural material", "diamond", ["hardest mineral", "hardest material"]),
    ("softest mineral mohs", "talc", ["softest mineral", "mohs 1"]),
    # History / culture (short)
    ("first us president", "George Washington", [
        "first president usa", "first american president",
    ]),
    ("penicillin discoverer", "Alexander Fleming", ["who discovered penicillin"]),
    ("relativity author", "Albert Einstein", [
        "who wrote relativity", "einstein theory",
    ]),
    ("origin of species author", "Charles Darwin", [
        "who wrote origin of species", "darwin book",
    ]),
    ("hamlet author", "William Shakespeare", ["who wrote hamlet"]),
    ("iliad author", "Homer", ["who wrote the iliad"]),
    ("odyssey author", "Homer", ["who wrote the odyssey"]),
    ("www inventor", "Tim Berners-Lee", [
        "who invented the web", "inventor of www", "inventor of the world wide web",
    ]),
]

# Wave 2 (2026-07-30): currencies, more geo, CS/tech acronyms, school science, animals.
WAVE2_ENTRIES: list[tuple[str, str, list[str]]] = [
    # Currencies
    ("currency of japan", "yen (JPY)", ["japan currency", "japanese yen", "currency japan"]),
    ("currency of uk", "pound sterling (GBP)", [
        "uk currency", "british pound", "currency britain", "currency england",
    ]),
    ("currency of usa", "US dollar (USD)", [
        "us currency", "american dollar", "currency usa", "currency united states",
    ]),
    ("currency of eurozone", "euro (EUR)", ["euro currency", "currency euro", "eu currency"]),
    ("currency of china", "yuan / renminbi (CNY)", [
        "china currency", "chinese yuan", "currency china", "rmb",
    ]),
    ("currency of india", "Indian rupee (INR)", ["india currency", "currency india"]),
    ("currency of canada", "Canadian dollar (CAD)", ["canada currency", "currency canada"]),
    ("currency of australia", "Australian dollar (AUD)", [
        "australia currency", "currency australia",
    ]),
    ("currency of switzerland", "Swiss franc (CHF)", [
        "switzerland currency", "currency switzerland",
    ]),
    ("currency of south korea", "won (KRW)", [
        "south korea currency", "korean won", "currency south korea",
    ]),
    ("currency of brazil", "real (BRL)", ["brazil currency", "currency brazil"]),
    ("currency of mexico", "peso (MXN)", ["mexico currency", "currency mexico"]),
    ("currency of russia", "ruble (RUB)", ["russia currency", "currency russia"]),
    ("currency of turkey", "lira (TRY)", ["turkey currency", "currency turkey"]),
    ("currency of south africa", "rand (ZAR)", [
        "south africa currency", "currency south africa",
    ]),
    # More capitals
    ("capital of peru", "Lima", ["peru capital", "capital peru"]),
    ("capital of colombia", "Bogota", ["colombia capital", "capital colombia"]),
    ("capital of venezuela", "Caracas", ["venezuela capital", "capital venezuela"]),
    ("capital of cuba", "Havana", ["cuba capital", "capital cuba"]),
    ("capital of morocco", "Rabat", ["morocco capital", "capital morocco"]),
    ("capital of ethiopia", "Addis Ababa", ["ethiopia capital", "capital ethiopia"]),
    ("capital of ghana", "Accra", ["ghana capital", "capital ghana"]),
    ("capital of tanzania", "Dodoma", ["tanzania capital", "capital tanzania"]),
    ("capital of czechia", "Prague", [
        "czech republic capital", "capital czech republic", "capital of czech republic",
        "capital czechia",
    ]),
    ("capital of hungary", "Budapest", ["hungary capital", "capital hungary"]),
    ("capital of romania", "Bucharest", ["romania capital", "capital romania"]),
    ("capital of bulgaria", "Sofia", ["bulgaria capital", "capital bulgaria"]),
    ("capital of serbia", "Belgrade", ["serbia capital", "capital serbia"]),
    ("capital of croatia", "Zagreb", ["croatia capital", "capital croatia"]),
    ("capital of iceland", "Reykjavik", ["iceland capital", "capital iceland"]),
    ("capital of myanmar", "Naypyidaw", ["myanmar capital", "capital myanmar", "burma capital"]),
    ("capital of cambodia", "Phnom Penh", ["cambodia capital", "capital cambodia"]),
    ("capital of laos", "Vientiane", ["laos capital", "capital laos"]),
    ("capital of nepal", "Kathmandu", ["nepal capital", "capital nepal"]),
    ("capital of sri lanka", "Sri Jayawardenepura Kotte (legislative); Colombo (commercial)", [
        "sri lanka capital", "capital sri lanka", "capital of colombo",
    ]),
    ("capital of afghanistan", "Kabul", ["afghanistan capital", "capital afghanistan"]),
    ("capital of uae", "Abu Dhabi", [
        "uae capital", "capital uae", "united arab emirates capital",
        "capital of united arab emirates",
    ]),
    ("capital of qatar", "Doha", ["qatar capital", "capital qatar"]),
    ("capital of kuwait", "Kuwait City", ["kuwait capital", "capital kuwait"]),
    ("capital of jordan", "Amman", ["jordan capital", "capital jordan"]),
    ("capital of lebanon", "Beirut", ["lebanon capital", "capital lebanon"]),
    ("capital of israel", "Jerusalem", ["israel capital", "capital israel"]),
    ("capital of palestine", "Ramallah (admin); East Jerusalem claimed", [
        "palestine capital", "capital palestine",
    ]),
    # Geo extremes
    ("largest country by area", "Russia", [
        "largest country", "biggest country", "largest country on earth",
    ]),
    ("smallest country by area", "Vatican City", [
        "smallest country", "smallest country on earth", "vatican size rank",
    ]),
    ("most populous country", "India (as of mid-2020s estimates; China close)", [
        "largest population country", "most people country", "most populated country",
    ]),
    ("longest river africa", "Nile", ["longest river in africa"]),
    ("longest river south america", "Amazon", ["longest river in south america"]),
    ("highest waterfall", "Angel Falls", ["tallest waterfall", "highest waterfall on earth"]),
    ("largest lake freshwater", "Lake Superior (by surface among pure freshwater; Baikal by volume)", [
        "largest freshwater lake",
    ]),
    ("largest desert hot", "Sahara", ["largest hot desert", "sahara size"]),
    ("coldest place inhabited", "Oymyakon region (often cited among coldest settlements)", [
        "coldest town", "coldest inhabited place",
    ]),
    # SI / physics school
    ("si unit force", "newton (N)", ["unit of force", "si force", "newton unit"]),
    ("si unit energy", "joule (J)", ["unit of energy", "si energy", "joule unit"]),
    ("si unit power", "watt (W)", ["unit of power", "si power", "watt unit"]),
    ("si unit pressure", "pascal (Pa)", ["unit of pressure", "si pressure"]),
    ("si unit frequency", "hertz (Hz)", ["unit of frequency", "si frequency"]),
    ("si unit electric charge", "coulomb (C)", ["unit of charge", "si charge"]),
    ("si unit voltage", "volt (V)", ["unit of voltage", "si voltage", "unit of potential difference"]),
    ("si unit resistance", "ohm", ["unit of resistance", "si resistance"]),
    ("ohms law", "V = I * R", ["ohm law", "ohm's law", "ohms law formula"]),
    ("newtons second law", "F = m * a", [
        "newton second law", "f=ma", "newtons 2nd law",
    ]),
    ("kinetic energy formula", "KE = 1/2 * m * v^2", [
        "formula for kinetic energy", "kinetic energy equation",
    ]),
    ("potential energy gravity", "PE = m * g * h", [
        "gravitational potential energy", "formula potential energy",
    ]),
    ("work formula", "W = F * d (force times distance along force)", [
        "formula for work physics", "work equals force times distance",
    ]),
    ("density formula", "density = mass / volume", ["formula for density"]),
    ("speed formula", "speed = distance / time", ["formula for speed", "velocity average formula"]),
    ("wave speed formula", "v = f * wavelength", ["wave equation speed"]),
    ("celsius absolute zero", "-273.15 C", ["absolute zero celsius"]),
    ("boiling point ethanol", "about 78 degrees Celsius", [
        "ethanol boiling point", "alcohol boiling point approx",
    ]),
    ("melting point iron", "about 1538 degrees Celsius", [
        "iron melting point", "melting point of iron",
    ]),
    ("melting point gold", "about 1064 degrees Celsius", ["gold melting point"]),
    ("atomic number carbon", "6", ["carbon atomic number", "atoms carbon number"]),
    ("atomic number oxygen", "8", ["oxygen atomic number"]),
    ("atomic number iron", "26", ["iron atomic number"]),
    ("atomic number gold", "79", ["gold atomic number"]),
    ("atomic number uranium", "92", ["uranium atomic number"]),
    ("water formula", "H2O", ["formula of water", "chemical formula water"]),
    ("carbon dioxide formula", "CO2", ["co2 formula", "formula carbon dioxide"]),
    ("salt formula", "NaCl", ["table salt formula", "sodium chloride formula"]),
    ("ammonia formula", "NH3", ["formula ammonia"]),
    ("methane formula", "CH4", ["formula methane", "natural gas main molecule"]),
    ("ozone formula", "O3", ["formula ozone"]),
    ("glucose formula", "C6H12O6", ["formula glucose", "sugar formula glucose"]),
    # Biology school
    ("mitochondria function", "produce ATP (cell powerhouse)", [
        "what do mitochondria do", "mitochondria role", "powerhouse of the cell",
    ]),
    ("ribosome function", "protein synthesis", [
        "what do ribosomes do", "ribosome role",
    ]),
    ("nucleus function", "holds DNA and controls the cell", [
        "what does the nucleus do", "cell nucleus role",
    ]),
    ("chloroplast function", "photosynthesis in plant cells", [
        "what do chloroplasts do", "chloroplast role",
    ]),
    ("number of bones human adult", "206", [
        "how many bones human", "human bones count", "bones in adult body",
    ]),
    ("number of teeth adult human", "32", [
        "how many teeth adult", "adult teeth count",
    ]),
    ("blood cells red function", "carry oxygen using hemoglobin", [
        "red blood cell function", "rbc function",
    ]),
    ("insulin function", "lowers blood glucose", [
        "what does insulin do", "insulin role",
    ]),
    ("vaccine definition", "trains immune system against a pathogen without full disease", [
        "what is a vaccine",
    ]),
    ("virus vs bacteria", "viruses need host cells; bacteria are cells that can live independently", [
        "difference virus bacteria", "virus or bacteria",
    ]),
    # Animals
    ("largest mammal", "blue whale", ["biggest mammal", "largest animal mammal"]),
    ("largest land animal", "African elephant", ["biggest land animal"]),
    ("fastest land animal", "cheetah", ["fastest animal on land"]),
    ("fastest bird dive", "peregrine falcon", ["fastest bird", "fastest animal dive"]),
    ("tallest animal", "giraffe", ["tallest land animal"]),
    ("largest bird", "ostrich (living)", ["biggest bird"]),
    ("only mammal that flies", "bat", ["flying mammal"]),
    ("marsupial example", "kangaroo (also koala, wallaby)", ["what is a marsupial example"]),
    # CS / tech
    ("cpu full form", "central processing unit", ["cpu stands for", "what is cpu"]),
    ("gpu full form", "graphics processing unit", ["gpu stands for", "what is gpu"]),
    ("ram full form", "random access memory", ["ram stands for", "what is ram"]),
    ("rom full form", "read-only memory", ["rom stands for"]),
    ("ssd full form", "solid-state drive", ["ssd stands for", "what is ssd"]),
    ("hdd full form", "hard disk drive", ["hdd stands for"]),
    ("usb full form", "universal serial bus", ["usb stands for"]),
    ("http full form", "hypertext transfer protocol", ["http stands for"]),
    ("https full form", "hypertext transfer protocol secure", ["https stands for"]),
    ("html full form", "hypertext markup language", ["html stands for"]),
    ("css full form", "cascading style sheets", ["css stands for"]),
    ("json full form", "javascript object notation", ["json stands for"]),
    ("xml full form", "extensible markup language", ["xml stands for"]),
    ("sql full form", "structured query language", ["sql stands for"]),
    ("api full form", "application programming interface", ["api stands for", "what is an api"]),
    ("url full form", "uniform resource locator", ["url stands for"]),
    ("ip full form", "internet protocol", ["ip stands for"]),
    ("dns full form", "domain name system", ["dns stands for"]),
    ("os full form", "operating system", ["os stands for"]),
    ("gui full form", "graphical user interface", ["gui stands for"]),
    ("cli full form", "command-line interface", ["cli stands for"]),
    ("ide full form", "integrated development environment", ["ide stands for"]),
    ("ai full form", "artificial intelligence", ["ai stands for"]),
    ("ml full form", "machine learning", ["ml stands for"]),
    ("nlp full form", "natural language processing", ["nlp stands for"]),
    ("http status 404", "Not Found", ["404 meaning", "what is 404", "http 404"]),
    ("http status 200", "OK", ["200 meaning", "http 200"]),
    ("http status 500", "Internal Server Error", ["500 meaning", "http 500"]),
    ("python creator", "Guido van Rossum", [
        "who created python", "inventor of python", "python author",
    ]),
    ("linux creator", "Linus Torvalds", [
        "who created linux", "inventor of linux", "linux author",
    ]),
    ("git creator", "Linus Torvalds", ["who created git", "git author"]),
    ("c language creators", "Dennis Ritchie (with Bell Labs Unix context)", [
        "who created c", "c language inventor",
    ]),
    ("java creator", "James Gosling", ["who created java", "java author"]),
    ("javascript creator", "Brendan Eich", ["who created javascript", "js creator"]),
    ("telephone inventor", "Alexander Graham Bell (patented; contested history)", [
        "who invented the telephone", "inventor of telephone",
    ]),
    ("light bulb popularizer", "Thomas Edison (commercial electric lighting; many prior inventors)", [
        "who invented the light bulb", "light bulb inventor",
    ]),
    ("airplane inventors", "Wright brothers (Orville and Wilbur)", [
        "who invented the airplane", "first airplane",
    ]),
    ("printing press europe", "Johannes Gutenberg (movable type in Europe)", [
        "who invented the printing press", "gutenberg press",
    ]),
    # Orgs / civics short
    ("un headquarters", "New York City", [
        "united nations headquarters", "where is un hq", "un hq",
    ]),
    ("nato full form", "North Atlantic Treaty Organization", [
        "nato stands for", "what is nato",
    ]),
    ("who full form", "World Health Organization", ["who stands for health", "world health org"]),
    ("nasa full form", "National Aeronautics and Space Administration", [
        "nasa stands for", "what is nasa",
    ]),
    ("eu full form", "European Union", ["eu stands for", "what is the eu"]),
    ("usa full form", "United States of America", ["usa stands for", "us full name"]),
    ("uk full form", "United Kingdom of Great Britain and Northern Ireland", [
        "uk stands for", "united kingdom full name",
    ]),
    ("leap year days", "366", ["days in leap year", "how many days leap year"]),
    ("moon phases main", "new, waxing, full, waning (cycle ~29.5 days)", [
        "phases of the moon", "moon phase cycle",
    ]),
    ("earth orbit sun period", "about 365.25 days (one year)", [
        "how long earth orbits sun", "earth year length",
    ]),
    ("earth rotates period", "about 24 hours (one day)", [
        "how long earth rotation", "length of day earth",
    ]),
    ("number of planets solar system", "8", [
        "how many planets", "planets in solar system",
    ]),
    ("planet order from sun", "Mercury Venus Earth Mars Jupiter Saturn Uranus Neptune", [
        "order of planets", "planets in order",
    ]),
    # Language / misc school
    ("english alphabet letters", "26", ["how many letters english", "letters in alphabet"]),
    ("pi approximate 2dp", "3.14", ["pi to 2 decimal places", "pi 2dp"]),
    ("pi approximate 5dp", "3.14159", ["pi to 5 decimals"]),
    ("percent means", "per hundred", ["what does percent mean", "meaning of percent"]),
    ("dozen", "12", ["how many in a dozen"]),
    ("score number", "20", ["how many in a score"]),
    ("fortnight", "14 days", ["how long is a fortnight"]),
    ("l3 cache meaning", "last-level cache on a CPU (shared per complex on many designs)", [
        "what is l3 cache", "cpu l3",
    ]),
    ("ternary weights meaning", "weights in {-1, 0, +1} (BitNet-style low-bit packs)", [
        "what is ternary weight", "ternary quantization",
    ]),
]

# Wave3: more SciQ/school science cards (host Engram; no weight growth)
WAVE3_ENTRIES: list[tuple[str, str, list[str]]] = [
    # Physics
    ("first law of thermodynamics", "energy cannot be created or destroyed, only transformed", [
        "1st law thermodynamics", "conservation of energy law",
    ]),
    ("second law of thermodynamics", "entropy of an isolated system tends to increase", [
        "2nd law thermodynamics", "entropy law",
    ]),
    ("third law of thermodynamics", "entropy approaches a minimum as temperature approaches absolute zero", [
        "3rd law thermodynamics",
    ]),
    ("newtons first law", "an object stays at rest or in uniform motion unless a net force acts", [
        "newton first law", "law of inertia", "newtons 1st law",
    ]),
    ("newtons third law", "for every action there is an equal and opposite reaction", [
        "newton third law", "newtons 3rd law", "action reaction law",
    ]),
    ("power formula physics", "P = W / t (work over time); also P = V * I for circuits", [
        "formula for power physics", "electrical power formula",
    ]),
    ("frequency unit", "hertz (Hz)", ["unit of frequency", "si unit frequency"]),
    ("force unit", "newton (N)", ["unit of force", "si unit force"]),
    ("energy unit", "joule (J)", ["unit of energy", "si unit energy"]),
    ("pressure unit si", "pascal (Pa)", ["unit of pressure si", "si pressure"]),
    ("charge unit", "coulomb (C)", ["unit of electric charge", "si charge"]),
    ("acceleration due to gravity", "about 9.8 m/s^2 on Earth", [
        "g earth", "earth gravity acceleration", "standard gravity",
    ]),
    ("light year definition", "distance light travels in one year (about 9.46e12 km)", [
        "what is a light year", "lightyear meaning",
    ]),
    ("reflection law", "angle of incidence equals angle of reflection", [
        "law of reflection", "mirror reflection rule",
    ]),
    ("refraction definition", "bending of light when it passes between media of different speed", [
        "what is refraction", "light refraction",
    ]),
    ("conduction heat", "heat transfer through direct contact of matter", [
        "heat conduction", "thermal conduction",
    ]),
    ("convection heat", "heat transfer by bulk motion of a fluid", [
        "heat convection", "thermal convection",
    ]),
    ("radiation heat", "heat transfer by electromagnetic waves (no medium required)", [
        "heat radiation", "thermal radiation",
    ]),
    # Chemistry
    ("atomic number hydrogen", "1", ["hydrogen atomic number"]),
    ("atomic number helium", "2", ["helium atomic number"]),
    ("atomic number nitrogen", "7", ["nitrogen atomic number"]),
    ("atomic number sodium", "11", ["sodium atomic number"]),
    ("atomic number chlorine", "17", ["chlorine atomic number"]),
    ("atomic number calcium", "20", ["calcium atomic number"]),
    ("proton charge", "positive (+1 elementary charge)", ["charge of proton"]),
    ("electron charge", "negative (-1 elementary charge)", ["charge of electron"]),
    ("neutron charge", "zero (neutral)", ["charge of neutron"]),
    ("isotope definition", "same element (same protons) different neutron count", [
        "what is an isotope", "isotopes meaning",
    ]),
    ("ion definition", "atom or molecule with a net electric charge", [
        "what is an ion", "ions meaning",
    ]),
    ("ph scale range", "typically 0 to 14 (7 neutral in water at 25 C)", [
        "ph scale", "range of ph",
    ]),
    ("oxidation definition school", "loss of electrons (or gain of oxygen in older school wording)", [
        "what is oxidation", "oxidation meaning",
    ]),
    ("reduction definition school", "gain of electrons (or loss of oxygen in older school wording)", [
        "what is reduction chemistry", "reduction meaning chemistry",
    ]),
    ("catalyst definition", "speeds a reaction without being permanently consumed", [
        "what is a catalyst", "catalyst meaning",
    ]),
    ("exothermic reaction", "releases heat to surroundings", [
        "what is exothermic", "exothermic meaning",
    ]),
    ("endothermic reaction", "absorbs heat from surroundings", [
        "what is endothermic", "endothermic meaning",
    ]),
    ("periodic table rows", "periods", ["what are periods periodic table"]),
    ("periodic table columns", "groups or families", [
        "what are groups periodic table", "periodic table families",
    ]),
    ("noble gases", "helium neon argon krypton xenon radon (group 18)", [
        "inert gases", "group 18 elements",
    ]),
    ("water polarity", "water is a polar molecule", [
        "is water polar", "polar water molecule",
    ]),
    # Biology
    ("cell theory", "all living things are made of cells; cells come from pre-existing cells", [
        "what is cell theory", "cell theory main points",
    ]),
    ("homeostasis definition", "maintaining stable internal conditions", [
        "what is homeostasis", "homeostasis meaning",
    ]),
    ("natural selection", "traits that aid survival/reproduction become more common over generations", [
        "what is natural selection", "darwin natural selection short",
    ]),
    ("gene definition", "a unit of heredity; a DNA segment that codes for a product", [
        "what is a gene", "gene meaning",
    ]),
    ("allele definition", "a variant form of a gene", ["what is an allele"]),
    ("mitosis purpose", "produce two genetically identical daughter cells for growth/repair", [
        "what is mitosis for", "mitosis function",
    ]),
    ("meiosis purpose", "produce gametes with half the chromosome number", [
        "what is meiosis for", "meiosis function",
    ]),
    ("ecosystem definition", "community of organisms plus their physical environment", [
        "what is an ecosystem",
    ]),
    ("food chain producer", "organism that makes its own food (usually photosynthetic plants/algae)", [
        "what is a producer ecology", "primary producer",
    ]),
    ("food chain consumer", "organism that eats other organisms for energy", [
        "what is a consumer ecology",
    ]),
    ("decomposer role", "break down dead matter and recycle nutrients", [
        "what do decomposers do", "decomposer function",
    ]),
    ("ventricle heart", "lower chambers that pump blood out of the heart", [
        "what are ventricles heart", "heart ventricles",
    ]),
    ("atrium heart", "upper chambers that receive blood into the heart", [
        "what are atria heart", "heart atria",
    ]),
    ("neuron function", "transmit electrical/chemical signals in the nervous system", [
        "what do neurons do", "nerve cell function",
    ]),
    ("enzyme definition", "biological catalyst (usually a protein)", [
        "what is an enzyme", "enzyme meaning",
    ]),
    # Earth / space school
    ("rock cycle main types", "igneous, sedimentary, metamorphic", [
        "three rock types", "types of rocks",
    ]),
    ("igneous rock", "forms from cooling magma or lava", [
        "what is igneous rock", "igneous meaning",
    ]),
    ("sedimentary rock", "forms from compacted sediments, often in layers", [
        "what is sedimentary rock",
    ]),
    ("metamorphic rock", "forms when existing rock is changed by heat/pressure", [
        "what is metamorphic rock",
    ]),
    ("water cycle main steps", "evaporation, condensation, precipitation, collection", [
        "hydrologic cycle steps", "steps of water cycle",
    ]),
    ("ozone layer role", "absorbs much of the Sun's harmful ultraviolet radiation", [
        "what does ozone layer do", "ozone layer function",
    ]),
    ("greenhouse effect", "atmosphere traps heat; keeps Earth warmer than without gases like CO2/H2O", [
        "what is greenhouse effect", "greenhouse effect short",
    ]),
    ("tectonic plates", "large moving pieces of Earth's lithosphere", [
        "what are tectonic plates", "plate tectonics short",
    ]),
    ("epicenter earthquake", "point on Earth's surface above the focus", [
        "what is epicenter", "earthquake epicenter",
    ]),
    ("galaxy type milky way", "barred spiral galaxy", [
        "what type of galaxy is milky way", "milky way galaxy type",
    ]),
    ("sun composition main", "mostly hydrogen and helium", [
        "what is the sun made of", "sun elements",
    ]),
    ("comet vs asteroid", "comets are icy and often show tails near the Sun; asteroids are rockier", [
        "difference comet asteroid", "comet or asteroid",
    ]),
]


ALL_NEW_ENTRIES: list[tuple[str, str, list[str]]] = (
    NEW_ENTRIES + WAVE2_ENTRIES + WAVE3_ENTRIES
)


def load_kb(path: Path) -> dict:
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict) or "entries" not in data:
        raise SystemExit(f"expected v2 kb with entries: {path}")
    return data


def rebuild_flat(entries: list[dict]) -> dict[str, str]:
    flat: dict[str, str] = {}
    conflicts = 0
    for ent in entries:
        val = str(ent.get("value", "")).strip()
        if not val:
            continue
        keys = [str(ent.get("id", ""))] + list(ent.get("aliases") or [])
        for raw in keys:
            k = norm_key(raw)
            if not k:
                continue
            if k in flat and flat[k] != val:
                conflicts += 1
                continue
            flat[k] = val
    if conflicts:
        print(f"warn: skipped {conflicts} conflicting key writes", file=sys.stderr)
    return flat


def existing_key_map(entries: list[dict]) -> dict[str, str]:
    return rebuild_flat(entries)


def merge_entry(
    entries: list[dict],
    eid: str,
    value: str,
    aliases: list[str],
    flat: dict[str, str],
) -> tuple[int, int]:
    """Add entry or merge aliases. Returns (new_entries, new_keys)."""
    eid_n = norm_key(eid)
    value = value.strip()
    if not eid_n or not value:
        return 0, 0
    if eid_n in flat and flat[eid_n] != value:
        return 0, 0  # refuse existing fact

    # find existing by id
    for ent in entries:
        if norm_key(str(ent.get("id", ""))) == eid_n:
            # same id: merge aliases only if values match
            if str(ent.get("value", "")).strip() != value:
                return 0, 0
            have = {norm_key(a) for a in (ent.get("aliases") or [])}
            added = 0
            for a in aliases:
                ak = norm_key(a)
                if not ak or ak == eid_n:
                    continue
                if ak in flat and flat[ak] != value:
                    continue
                if ak not in have:
                    ent.setdefault("aliases", []).append(a)
                    have.add(ak)
                    flat[ak] = value
                    added += 1
            return 0, added

    # brand new entry
    clean_aliases: list[str] = []
    for a in aliases:
        ak = norm_key(a)
        if not ak or ak == eid_n:
            continue
        if ak in flat and flat[ak] != value:
            continue
        if ak not in {norm_key(x) for x in clean_aliases}:
            clean_aliases.append(a)
    entries.append({"id": eid, "value": value, "aliases": clean_aliases})
    flat[eid_n] = value
    for a in clean_aliases:
        flat[norm_key(a)] = value
    return 1, 1 + len(clean_aliases)


def enrich_capital_aliases(entries: list[dict], flat: dict[str, str]) -> int:
    """If 'capital of X' exists, add 'capital X' / 'X capital' when free."""
    added = 0
    for ent in entries:
        eid = norm_key(str(ent.get("id", "")))
        val = str(ent.get("value", "")).strip()
        if not eid.startswith("capital of ") or not val:
            continue
        place = eid[len("capital of ") :].strip()
        if not place:
            continue
        extras = [f"capital {place}", f"{place} capital"]
        have = {norm_key(a) for a in (ent.get("aliases") or [])}
        for a in extras:
            ak = norm_key(a)
            if ak == eid or ak in have:
                continue
            if ak in flat and flat[ak] != val:
                continue
            ent.setdefault("aliases", []).append(a)
            have.add(ak)
            flat[ak] = val
            added += 1
    return added


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--path", type=Path, default=KB_PATH)
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    data = load_kb(args.path)
    entries: list[dict] = list(data.get("entries") or [])
    before_e = len(entries)
    flat = existing_key_map(entries)
    before_k = len(flat)

    new_e = new_k = 0
    for eid, val, aliases in ALL_NEW_ENTRIES:
        de, dk = merge_entry(entries, eid, val, aliases, flat)
        new_e += de
        new_k += dk

    alias_boost = enrich_capital_aliases(entries, flat)
    # Common "what is X" surface forms for short ids / acronyms
    what_boost = 0
    for ent in entries:
        eid = norm_key(str(ent.get("id", "")))
        val = str(ent.get("value", "")).strip()
        if not eid or not val:
            continue
        extras: list[str] = []
        if eid.endswith(" full form") or eid.endswith(" full name"):
            stem = eid.replace(" full form", "").replace(" full name", "").strip()
            if stem:
                extras += [f"what is {stem}", f"{stem} means", f"{stem} definition"]
        if eid.startswith("capital of "):
            place = eid[len("capital of ") :]
            extras += [f"what is the capital of {place}", f"what's the capital of {place}"]
        have = {norm_key(a) for a in (ent.get("aliases") or [])}
        for a in extras:
            ak = norm_key(a)
            if not ak or ak == eid or ak in have:
                continue
            if ak in flat and flat[ak] != val:
                continue
            ent.setdefault("aliases", []).append(a)
            have.add(ak)
            flat[ak] = val
            what_boost += 1
    alias_boost += what_boost

    # final rebuild (canonical)
    flat = rebuild_flat(entries)
    after_e, after_k = len(entries), len(flat)

    note = (
        f"Engram-lite expanded {date.today().isoformat()} "
        f"(+{after_e - before_e} entries, keys {before_k}->{after_k})"
    )
    out = {
        "version": 2,
        "note": note,
        "n_entries": after_e,
        "n_keys": after_k,
        "entries": entries,
        "flat": flat,
    }

    print(
        f"entries {before_e} -> {after_e} (+{new_e} new) | "
        f"keys {before_k} -> {after_k} "
        f"(+{after_k - before_k}; capital aliases +{alias_boost})"
    )
    if args.dry_run:
        print("dry-run: not writing")
        return 0

    args.path.write_text(json.dumps(out, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"wrote {args.path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
