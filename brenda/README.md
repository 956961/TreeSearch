# JsonSearcher based on BRENDA database
The BRENDA database is a comprehensive enzyme information system with a data structure that includes several main sections: Nomenclature, which covers systematic name, recommended name, synonyms, reaction, and reaction type; Organism, which includes protein ID, description, UniProt ID, commentary, reference ID, source tissue, and localization; Parameters, such as Km value, turnover number, Ki value, Pi value, optimum pH, pH range, optimum temperature, temperature range, specific activity, and IC50 value; Interactions, involving substrate and product, natural substrate and product, cofactor, metals and ions, inhibitors, and activating compounds; Molecular, which includes stability (such as general stability, storage stability, pH stability, organic solvent stability, oxidation stability, and temperature stability) and purification (such as cloned, engineering, renatured, and application); Structure, including molecular weight, subunits, posttranslational modification, and crystallization; and Bibliography, which includes reference ID, title, and PubMed ID.

## 1. Data preprocessing
### 1.1 Download data files

Download the data file from the website: www.brenda-enzymes.org/download.php (brenda_2023_1.json and brenda_2023_10.txt)

### 1.2 Complete the information
> The contents are organised in 40 information fields as given
below. Protein information is included in '#'...#', literature
citations are in '<...>', commentaries in '(...)' and field-
special information in '{...}'.

| Field | Information |
| --- | --- |
| /// | indicates the end of an EC-number specific part.  |	
|AC	|activating compound|
|AP	|application|
|CF	|cofactor|
|CL	|cloned
|CR	|crystallization|
|EN	|engineering|
|EXP|	expression|
|GI	|general information on enzyme|
|GS	|general stability|
|IC50|	IC-50 Value|
|ID	|EC-class|
|IN	|inhibitors|
|KKM|	Kcat/KM-Value substrate in {...}|
|KI	|Ki-value	inhibitor in {...}|
|KM	|KM-value	substrate in {...}|
|LO	|localization|
|ME	|metals/ions|
|MW	|molecular weight|
|NSP|	natural substrates/products	reversibilty information in {...}|
|OS	|oxygen stability|
|OSS|	organic solvent stability|
|PHO|	pH-optimum|
|PHR|	pH-range|
|PHS|	pH stability|
|PI|	isoelectric point|
|PM	|posttranslation modification|
|PR	|protein|
|PU	|purification|
|RE	|reaction catalyzed|
|RF	|references|
|REN|	renatured|
|RN	|accepted name (IUPAC) |
|RT	|reaction type|
|SA	|specific activity|
|SN	|synonyms|
|SP	|substrates/products	reversibilty information in {...}|
|SS	|storage stability|
|ST	|source/tissue|
|SU	|subunits|
|SY	|systematic name |
|TN	|turnover number	substrate in {...}|
|TO	|temperature optimum|
|TR	|temperature range|
|TS	|temperature	stability|
 
The protein information and field-special information are included in brenda_2023_10.txt. We extract it to complete the JSON database. The details can be found in the preprocess.ipynb.

### 1.3 Reorganize the JSON

The raw index in brenda_2023_1.json are the ids of different enzymes: "spontaneous', '1.1.1.1', '1.1.1.10', '1.1.1.100',...".

Each enzyme include 21 keys: \['id', 'name', 'systematic_name', 'synonyms', 'cofactor', 'inhibitor', 'subunits', 'reaction_type', 'purification', 'general_stability', 'storage_stability', 'generic_reaction', 'natural_reaction', 'reaction', 'km_value', 'ph_range', 'specific_activity', 'molecular_weight', 'organisms', 'references', 'proteins'\]

The value of these keys are still a json, for example: data\['organisms'\]\['0'\]: {'1': {'comment': '#1# SULT1A2 <1,2,6,7>', 'value': 'Cavia porcellus'}, '2': {'value': 'Mus musculus'}, '3': {'value': 'Homo sapiens'}, '4': {'value': 'Rattus norvegicus'}}. For ease of lookup, we change the json format to the list. The details can be found in the preprocess.ipynb.

After the preprocessing, we can save the database as .csv or .json format.

## 2. Create the JSON agent

Based on langchain and our JSON file, we can create our own JSON agent. At First, we need to deploy a large language model locally or use the Openai api. Here, we use 'Llama-2-7b-chat' as our local choice. We also use gpt-3.5-turbo as an alternative if 'Llama-2-7b-chat' can not give a satisfactory answer. The details can be found in the JSONSearcher.py.

Sometimes, JSON agent is prone to keyerror issues. As an alternative, we can change the data format to .csv and create a table agent which is in the TableSearcher.py.


### Here are examples of asking questions through the JSON agent:

### Q: What is the optimal temperature range of all-trans-retinol dehydrogenase (NAD+)?

** > Entering new AgentExecutor chain...
Action: json_spec_list_keys
Action Input: data
Observation: ['id', 'systematic_name', 'synonyms', 'cofactor', 'inhibitor', 'subunits', 'reaction_type', 'purification', 'general_stability', 'storage_stability', 'generic_reaction', 'natural_reaction', 'reaction', 'km_value', 'ph_range', 'temperature', 'molecular_weight', 'organisms', 'references', 'proteins']
Thought:I should check the temperature key to find the optimal temperature range for the enzyme.
Action: json_spec_get_value
Action Input: data["temperature"]
Observation: {'spontaneous reaction': 'none', 'alcohol dehydrogenase': '#10# isoenzyme 4 <49>', 'L-xylulose reductase': '#12# xylitol dehydrogenase activity <4>', '3-oxoacyl-[acyl-carrier-protein] reductase': 'none', 'acylglycerone-phosphate reductase': 'none', '3-dehydrosphinganine reductase': 'none', 'L-threonine 3-dehydrogenase': 'none', '4-oxoproline reductase': 'none', 'all-trans-retinol dehydrogenase (NAD+)': '#17# pH 6.5, 60°C <25>; #23# with all-trans-retinal as substrate, at pH 6.5 and 60°C <25>', 'pantoate 4-dehydrogenase': 'none'}
Thought:The optimal temperature range for all-trans-retinol dehydrogenase (NAD+) is 60°C.
Final Answer: 60°C

> Finished chain.
60°C

### Q: What is the catalytic efficiency (kcat/Km) of L-threonine 3-dehydrogenase?

** > Entering new AgentExecutor chain...
Action: json_spec_list_keys
Action Input: data
Observation: ['id', 'name', 'systematic_name', 'synonyms', 'cofactor', 'inhibitor', 'subunits', 'reaction_type', 'purification', 'general_stability', 'storage_stability', 'generic_reaction', 'natural_reaction', 'reaction', 'km_value', 'ph_range', 'specific_activity', 'molecular_weight', 'organisms', 'references', 'proteins']
Thought:I should check the km_value key to see the catalytic efficiency of L-threonine 3-dehydrogenase.
Action: json_spec_get_value
Action Input: data["km_value"]
Observation: {'0': 'none', '1': '0.05', '2': '0.29', '3': '0.29', '4': '0.05', '5': '0.003', '6': '0.98', '7': '0.6', '8': '0.0015', '9': '0.078'}
Thought:The catalytic efficiency (km) of L-threonine 3-dehydrogenase is 0.078. 

Final Answer: 0.078

> Finished chain.
0.078

### Q: What is the pH stability range of glutamate racemase?

** > Entering new AgentExecutor chain...
Action: json_spec_list_keys
Action Input: data
Observation: ['id', 'name', 'systematic_name', 'synonyms', 'cofactor', 'inhibitor', 'subunits', 'reaction_type', 'purification', 'general_stability', 'storage_stability', 'generic_reaction', 'natural_reaction', 'reaction', 'km_value', 'ph_range', 'specific_activity', 'molecular_weight', 'organisms', 'references', 'proteins']
Thought:I should check the `ph_range` key to see the pH stability range of glutamate racemase.
Action: json_spec_get_value
Action Input: data["ph_range"]
Observation: {'0': 'none', '1': '4.0-7.5', '2': '6.0-8.0', '3': '5.5-9.5', '4': '5.5-8.2', '5': 'none', '6': '7.4-9.0', '7': 'none', '8': '7.5-8.0', '9': '7.0-10.0'}
Thought:I have found the pH stability range of glutamate racemase. Final Answer: The optimal pH range of glutamate racemase is 6.0-8.0.

> Finished chain.
The optimal pH range of glutamate racemase is 6.0-8.0.

### Q: What are the enzyme inhibition kinetics of L-threonine 3-dehydrogenase?

** > Entering new AgentExecutor chain...
Action: json_spec_list_keys
Action Input: data
Observation: ['id', 'systematic_name', 'synonyms', 'cofactor', 'inhibitor', 'subunits', 'reaction_type', 'purification', 'general_stability', 'storage_stability', 'generic_reaction', 'natural_reaction', 'reaction', 'km_value', 'ph_range', 'specific_activity', 'molecular_weight', 'organisms', 'references', 'proteins']
Thought:I should check the "inhibitor" key to see the enzyme inhibitor of L-threonine 3-dehydrogenase.
Action: json_spec_get_value
Action Input: data["inhibitor"][0]
Observation: KeyError(0)
Thought:I should try to access the "inhibitor" key directly without the index.
Action: json_spec_get_value
Action Input: data["inhibitor"]
Observation: {'spontaneous reaction': 'none', 'alcohol dehydrogenase': 'pefabloc:#103# 10 mM, 32% inhibition <173>', 'L-xylulose reductase': 'none', '3-oxoacyl-[acyl-carrier-protein] reductase': 'Tannic acid:#3,25# displays very strong inhibition <48>; #5# strongest inhibition on FabG, shows time-dependent irreversible inhibition <48>', 'acylglycerone-phosphate reductase': 'NADP+:#1# competitive to NADPH <5>', '3-dehydrosphinganine reductase': 'siRNA:#2# reduces expression of transfected FVT1 mRNA and protein by at least 70% <8>', 'L-threonine 3-dehydrogenase': 'Cd2+:#1# 0.05 and 1.0 mM, 90% inhibition <17>', '4-oxoproline reductase': 'none', 'all-trans-retinol dehydrogenase (NAD+)': 'imidazole:#12# increasing the amount of imidazole in the reaction mixture with purified RDH-E2 from 25 to 225mM significantly decreases RDH-E2 activity <18>', 'pantoate 4-dehydrogenase': 'p-chloromercuribenzoate:#1# 0.1 mM, 85% inhibition after 15 min, 200 mM 2-mercaptoethanol restores 70% of enzyme activity within 10 min, 20 mM D-pantoate and 1 mM NAD+ prevent inactivation when added simultaneously <2>'}
Thought:The enzyme inhibitor of L-threonine 3-dehydrogenase is Cd2+ at concentrations of 0.05 and 1.0 mM, resulting in 90% inhibition. 
Final Answer: Cd2+

> Finished chain.
Cd2+