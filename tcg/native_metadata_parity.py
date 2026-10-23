from collections import defaultdict
import ctypes
from hashlib import sha256
import json
from pathlib import Path
import re
import subprocess

from tcg.live_parity import LIVE_PINS, digest, read_pinned

NATIVE_METADATA_MANIFEST_SHA256="956b95ca89bcc7f771622c7e5f5507df9a0421eabbc5d365d35017ad054394b2"
NATIVE_METADATA_LABEL_MANIFEST_SHA256="0d2ac3aa86b0894c3817aab6499ccddcb0e9e1f10489318d298cb0b40903ace8"
NATIVE_METADATA_C13_MANIFEST_SHA256="9fec35b535681573cfa518a6b7e0bc9465d713a5dd1c157546ebfd4dca9a7cb1"
NATIVE_METADATA_TRAINER_GRAPH_MANIFEST_SHA256="1b8f61650d7602f26efac478e13df08faaf3f3157015df13a563cf1e701d8fec"
NATIVE_METADATA_TRAINER_CANONICAL_SHA256="3c156f5838b5a0a8d766c92a32f5d494a56a79d4338c5e7c1f1622e5e63a4339"
NATIVE_METADATA_TRAINER_GRAPH_SHA256="928f60626f1fcb00532625d44ef15a9c41df8cb1a413625a19469b9e7e49e42e"
NATIVE_METADATA_DOMAINS={
    NATIVE_METADATA_MANIFEST_SHA256:{"rows":18283,"programs":7184,"labelPolicy":False,
                                   "nativeCounts":{"cards":10215,"attacks":14255,"skills":2647}},
    NATIVE_METADATA_LABEL_MANIFEST_SHA256:{"rows":18948,"programs":7354,"labelPolicy":True,
                                         "nativeCounts":{"cards":10215,"attacks":14255,"skills":2647}},
    NATIVE_METADATA_C13_MANIFEST_SHA256:{"rows":18957,"programs":7356,"labelPolicy":True,"inventoryExtension":True,
                                       "nativeCounts":{"cards":10216,"attacks":14255,"skills":2648}},
    NATIVE_METADATA_TRAINER_GRAPH_MANIFEST_SHA256:{"rows":19056,"programs":7363,"labelPolicy":True,
        "inventoryExtension":True,"graphIdentityExtension":True,"nativeCounts":{"cards":10216,"attacks":14255,"skills":2648}},
}
NATIVE_METADATA_LABEL_POLICY={"id":"english-metadata-labels/1","replace":{"\u2019":"'"},"trimCharacters":" \t\r\n","fields":["name","evolvesFrom","evolvesFromBase[]","attacks[].name","abilities[].name","cardFunctions[]"],"preserved":"Exact raw labels/source bytes, non-label fields, source IDs/program hashes and physical native IDs."}
NATIVE_METADATA_PROBE_SOURCES={"probe.cpp":"5348b2bf0d91cf8c7d6ff553684391495835a1206063974f22b7d368e160e067","probe.vcxproj":"c4c94aca7640b35a0783e87ee296721f70dc3fbc14143a24b12494b5a9c86039"}
NATIVE_METADATA_HELD_PROGRAM_IDS={"2bce9505cc36589424658770d8232ea2399fe7172c42b088816478bd54821ef8":1426}

ENERGY_BITS={0:"C",1:"G",2:"R",4:"W",8:"L",16:"P",32:"F",64:"D",128:"M",256:"N",512:"Y"}
SOURCE_TYPES={1:3,2:2,4:4,8:8,16:11,32:5,64:7,128:10,256:9,512:6}
OWNER_FLAGS=("hop","lillie","iono","n","ethan","cynthia","misty","arven","steven","marnie","erika","larry")
REGIONAL_PROPERTIES={"Alolan":4,"Galarian":11,"Hisuian":17,"Paldean":21}
PROPERTY_FLAGS={"prismStar":5,"singleStrike":12,"rapidStrike":13,"fusionStrike":15,"radiant":18,"ancient":22,"future":23,"aceSpec":24,"teamPlasma":27,"ultraBeast":8,"tagTeam":2}
SUPPORTED_PROPERTIES={0,1,2,3,4,5,6,7,8,9,10,11,12,13,15,16,17,18,19,20,21,22,23,24,25,26,27}

class HeldMetadata(ValueError):pass

def require(value,reason):
    if not value:raise HeldMetadata(reason)


def metadata_label_projection(projection,policy):
    if policy is None:return projection
    require(policy==NATIVE_METADATA_LABEL_POLICY,"unrecognized English metadata label policy")
    def label(value):return value.replace("\u2019","'").strip(" \t\r\n")
    result=dict(projection)
    for field in ("name","evolvesFrom"):result[field]=label(result[field])
    result["evolvesFromBase"]=[label(value) for value in result["evolvesFromBase"]]
    for field in ("attacks","abilities"):result[field]=[{**item,"name":label(item["name"])} for item in result[field]]
    if "cardFunctions" in result:result["cardFunctions"]=[label(value) for value in result["cardFunctions"]]
    return result

def types(mask,present=True):
    require(type(mask) is int and mask>=0 and mask&~1023==0,"unknown_native_energy_mask")
    if not present:return []
    return sorted(v for bit,v in SOURCE_TYPES.items() if mask&bit) if mask else [1]

def printed_ability(name,vstar=False):
    require(isinstance(name,str) and name and name!="<empty>","missing_printed_ability_name")
    if name.startswith("[Ability] "):return {"name":name[len("[Ability] "):],"vstarPower":False}
    if name.startswith("[VSTAR Power] "):return {"name":name[len("[VSTAR Power] "):],"vstarPower":True}
    require(not name.startswith("["),"unsupported_printed_skill_kind")
    return {"name":name,"vstarPower":vstar}

def source_projection(row,passive):
    if row["format"]!=3:return source_non_pokemon(row,passive)
    require(row["stage"] in (1,2,3,4,5,6,8),"unsupported_source_stage")
    require(set(row["properties"])<=SUPPORTED_PROPERTIES,"unrepresented_source_property")
    require(all(value in ({}, {"conditions":[],"value":""}) for value in row["energyValues"]),
            "pokemon_energy_values_not_representable")
    attacks=[];abilities=[]
    printed_names={action["name"] for action in row["useActions"]+passive+row["triggeredActions"] if action["type"] in (1,2)}
    for channel,actions in (("useActions",row["useActions"]),("passiveActions",passive),("triggeredActions",row["triggeredActions"])):
        for action in actions:
            if channel=="useActions" and action["type"]==1:
                cost=action.get("cost","")
                require(isinstance(cost,str) and re.fullmatch(r"(?:\{[CGRWLPFDMNY]\})*",cost) is not None,"unsupported_source_attack_cost")
                attacks.append({"name":action["name"],"cost":cost})
            else:
                if channel!="useActions" and action["type"]==4 and action["name"] in printed_names|{"<empty>"}:continue
                require(action["type"]==2,"unrepresented_source_action_kind")
                require(not action.get("cost",""),"ability_energy_cost_not_representable")
                ability=printed_ability(action.get("name"))
                if ability not in abilities:abilities.append(ability)
    return {"name":row["name"],"format":row["format"],"stage":row["stage"],"hp":row["hp"],"prizeValue":row["prizeValue"],"properties":sorted(row["properties"]),"typing":sorted(row["typing"]),"weakness":sorted(row["weakness"]),"weaknessAmount":row["weaknessAmount"],"resistance":sorted(row["resistance"]),"resistanceAmount":row["resistanceAmount"],"retreat":row["retreat"],"evolvesFrom":row["evolvesFrom"],"evolvesFromBase":sorted(row["evolvesFromBase"]),"energyValues":[],"attacks":attacks,"abilities":abilities}

def native_projection(row,by_internal):
    if row["cardType"]!=0:return native_non_pokemon(row)
    flags=row["flags"];subtype=row["pokemonType"]
    require(not any(flags[k] for k in ("pokemonStar","deltaEvolution","pokemonSp","babyPokemon")),"unrepresented_native_trait")
    properties={1:[],3:[19],4:[19,26],5:[9],6:[10],7:[1],8:[16],9:[3,7],12:[3],13:[3,7],14:[6]}.get(subtype)
    require(properties is not None,"unsupported_native_pokemon_subtype")
    properties=list(properties)
    properties.extend(prop for region,prop in REGIONAL_PROPERTIES.items() if re.search(r"\b"+region+" ",row["name"]))
    for flag,prop in PROPERTY_FLAGS.items():
        if flags[flag]:properties.append(prop)
    if row["tera"]:properties.append(20)
    if any(flags[k] for k in OWNER_FLAGS):properties.append(25)
    properties=sorted(set(properties)) or [0]
    stage={1:1,2:2,3:3,4:6,5:8,6:4,8:4,9:5}.get(row["evolutionType"])
    require(stage is not None,"unsupported_native_evolution_type")
    def bases(card,seen):
        require(card["cardId"] not in seen,"native_parent_cycle")
        if card["evolutionType"]==1:return {card["name"]}
        parents=by_internal.get(card["parentInternal"],[])
        require(parents,"native_base_parent_not_representable")
        found=set()
        for parent in parents:found.update(bases(parent,seen|{card["cardId"]}))
        return found
    base=sorted(bases(row,set()))
    abilities=[]
    require(row["ancientTrait"] is None,"ancient_trait_metadata_not_represented")
    require(row["play"] is None,"pokemon_play_skill_not_represented")
    ability=row["ability"]
    if ability is not None:
        require(isinstance(ability,dict) and ability["skillType"]==0 and not ability["legacyPowerBody"],"unrepresented_native_printed_skill")
        abilities.append(printed_ability(ability["name"],ability["vstarPower"]))
    attacks=[]
    for attack in row["attacks"]:
        require(all(e in ENERGY_BITS for e in attack["energies"]),"unrepresented_native_attack_cost")
        attacks.append({"name":attack["name"],"cost":"".join("{"+ENERGY_BITS[e]+"}" for e in attack["energies"])})
    require(row["energyCount"]==0,"pokemon_energy_count_not_representable")
    return {"name":row["name"],"format":3,"stage":stage,"hp":row["hp"],"prizeValue":row["baselinePrizeValue"],"properties":properties,"typing":types(row["energyMask"]),"weakness":types(row["weaknessMask"],flags["hasWeakness"]),"weaknessAmount":(row["weaknessAddAmount"] or 2) if flags["hasWeakness"] else 0,"resistance":types(row["resistanceMask"],flags["hasResistance"]),"resistanceAmount":row["resistanceAmount"] if flags["hasResistance"] else 0,"retreat":row["retreat"],"evolvesFrom":row["parent"],"evolvesFromBase":base,"energyValues":[],"attacks":attacks,"abilities":abilities}

def internal_catalog(rows):
    result=defaultdict(list)
    for row in rows:result[row["internalName"]].append(row)
    return result

def source_non_pokemon(row,passive):
    form=row["format"]
    require(form in (1,2,4,5,6,7,8),"unsupported_source_format")
    require(set(row["properties"])<=SUPPORTED_PROPERTIES,"unrepresented_source_property")
    require(row["stage"]==0 and not row["evolvesFrom"],"non_pokemon_evolution_not_representable")
    functions=[];abilities=[];attacks=[]
    for channel,actions in (("useActions",row["useActions"]),("passiveActions",passive),("triggeredActions",row["triggeredActions"])):
        for action in actions:
            if channel=="useActions" and form==7 and action["type"]==1:
                cost=action.get("cost","")
                require(isinstance(cost,str) and re.fullmatch(r"(?:\{[CGRWLPFDMNY]\})*",cost) is not None,"unsupported_source_attack_cost")
                attacks.append({"name":action["name"],"cost":cost})
                continue
            require(not action.get("cost",""),"non_pokemon_action_energy_cost")
            if action["type"]==2:
                ability=printed_ability(action.get("name"))
                if ability not in abilities:abilities.append(ability)
            else:
                allowed=(9,) if form in (1,2) else (8,4) if form==7 else (3,4) if form==8 else (3,)
                require(action["type"] in allowed and action.get("name")==row["name"],"unrepresented_source_card_function")
                if row["name"] not in functions:functions.append(row["name"])
    energy=[]
    for value in row["energyValues"]:
        require(not value.get("conditions",[]) and isinstance(value["value"],str) and re.fullmatch(r"(?:\{[CGRWLPFDMNY]\})+",value["value"]) is not None,"conditional_or_unrepresented_energy_value")
        energy.append(value["value"])
    require(len(energy)<=1,"multiple_energy_values_not_representable")
    require(form in (1,2) or not energy,"trainer_energy_values_not_representable")
    return {"name":row["name"],"format":form,"stage":0,"hp":row["hp"],"prizeValue":row["prizeValue"],"properties":sorted(row["properties"]),"typing":sorted(row["typing"]),"weakness":sorted(row["weakness"]),"weaknessAmount":row["weaknessAmount"],"resistance":sorted(row["resistance"]),"resistanceAmount":row["resistanceAmount"],"retreat":row["retreat"],"evolvesFrom":"","evolvesFromBase":[],"energyValues":energy,"attacks":attacks,"abilities":abilities,"cardFunctions":functions}

def native_non_pokemon(row):
    form={1:8 if row["pokemonType"] else 4,2:6,3:5,4:7,5:1,6:2}.get(row["cardType"])
    require(form is not None,"unsupported_native_format")
    flags=row["flags"]
    require(row["evolutionType"] in (0,1) and not row["parentInternal"] and not row["baseInternal"],"non_pokemon_native_evolution_not_representable")
    require(row["ancientTrait"] is None,"ancient_trait_metadata_not_represented")
    require(form==7 or not row["attacks"],"non_pokemon_attacks_not_representable")
    attacks=[]
    for attack in row["attacks"]:
        require(all(e in ENERGY_BITS for e in attack["energies"]),"unrepresented_native_attack_cost")
        attacks.append({"name":attack["name"],"cost":"".join("{"+ENERGY_BITS[e]+"}" for e in attack["energies"])})
    properties=sorted({prop for flag,prop in PROPERTY_FLAGS.items() if flags[flag]}|({25} if any(flags[k] for k in OWNER_FLAGS) else set())) or [0]
    functions=[];abilities=[]
    for channel in ("ability","play"):
        skill=row[channel]
        if skill is None:continue
        require(not skill["legacyPowerBody"] and not skill["vstarPower"] or skill["mainAbility"],"unrepresented_native_printed_skill")
        if skill["mainAbility"]:
            abilities.append(printed_ability(skill["name"],skill["vstarPower"]))
        else:
            require(skill["name"]==row["name"] and skill["skillType"] in (0,1,2),"unrepresented_native_card_function")
            if row["name"] not in functions:functions.append(row["name"])
    energy=[];typing=[]
    if form in (1,2):
        require(row["energyMask"] in ENERGY_BITS and row["energyCount"]>0,"native_energy_value_not_representable")
        energy=[("{"+ENERGY_BITS[row["energyMask"]]+"}")*row["energyCount"]]
        typing=types(row["energyMask"])*row["energyCount"]
    elif form==8:typing=types(row["energyMask"])
    else:require(row["energyCount"]==0 and row["energyMask"]==0,"trainer_native_energy_not_representable")
    return {"name":row["name"],"format":form,"stage":0,"hp":row["hp"],"prizeValue":row["baselinePrizeValue"] if form==8 else 0,"properties":properties,"typing":typing,"weakness":types(row["weaknessMask"],flags["hasWeakness"]),"weaknessAmount":(row["weaknessAddAmount"] or 2) if flags["hasWeakness"] else 0,"resistance":types(row["resistanceMask"],flags["hasResistance"]),"resistanceAmount":row["resistanceAmount"] if flags["hasResistance"] else 0,"retreat":row["retreat"],"evolvesFrom":"","evolvesFromBase":[],"energyValues":energy,"attacks":attacks,"abilities":abilities,"cardFunctions":functions}


def source_metadata(card):
    def action(value,channel):
        if channel=="passiveActions":name=value.get("EffectName","<empty>")
        else:name=value.get("actionName","<empty>")
        default={(4,"useActions"):3,(5,"useActions"):3,(4,"triggeredActions"):3,(6,"triggeredActions"):3}.get((card.get("cardFormat",3),channel),2 if channel!="useActions" else 0)
        return {"name":name,"type":value.get("actionType",default),"cost":value.get("energyCost","")}
    return {"name":card["cardName"],"format":card.get("cardFormat",3),"stage":card.get("cardStage",1),"hp":card.get("hp",0),"prizeValue":card.get("prizeValue",0),"properties":card.get("cardProperties",[]),"typing":card.get("cardTyping",[]),"weakness":card.get("weakness",[]),"weaknessAmount":card.get("weaknessAmt",0),"resistance":card.get("resistance",[]),"resistanceAmount":card.get("resistanceAmt",0),"retreat":card.get("retreatCost",0),"evolvesFrom":card.get("evolvesFrom",""),"evolvesFromBase":card.get("evolvesFromBase",[]),"energyValues":card.get("energyValues",[]),"useActions":[action(a,"useActions") for a in card.get("useActions",[])],"passiveActions":[action(a,"passiveActions") for a in card.get("passiveActions",[])],"triggeredActions":[action(a,"triggeredActions") for a in card.get("triggeredActions",[])]}


def native_metadata_scenario(item):
    return {"suite":"native-full-metadata/1","commands":[{"operation":"nativeCardTableMetadata","cardSourceID":item["cardSourceID"],"cardSha256":item["cardSha256"],"programSha256":item["programSha256"]}]}


def verify_metadata_inventory_extension(manifest,receipt,items,raw,registry,skills):
    extension=manifest["inventoryExtension"]
    require(extension["schema"]=="native-metadata-inventory-extension/1"
            and extension["parentManifestSha256"]==NATIVE_METADATA_LABEL_MANIFEST_SHA256
            and extension["engineSourceCommit"]==receipt["engineSourceCommit"]
            and extension["librarySha256"]==receipt["librarySha256"]
            and extension["oldExportedPhysicalIdentitiesUnchanged"] is True,"native metadata additive inventory provenance changed")
    parent=json.loads(read_pinned(extension["parentManifestPath"],extension["parentManifestSha256"]))
    old={name:json.loads(read_pinned(binding["path"],binding["sha256"]))
         for name,binding in extension["parentArtifacts"].items()}
    tables=((old["native-raw.json"],raw,"cardId"),
            (old["registry.json"]["entries"],registry["entries"],"cardId"),
            (old["registry.json"]["attacks"],registry["attacks"],"attackId"),
            (old["native-skills.json"],skills,"skillId"))
    for before,after,field in tables:
        actual={row[field]:row for row in after}
        require(len(actual)==len(after) and all(actual.get(row[field])==row for row in before),
                "native metadata extension removed or changed an old physical inventory record")
    anchors=manifest["mechanicsProgramAnchors"]
    require(len(anchors)==1,"native metadata extension has an unreviewed mechanics selection")
    for program,anchor in anchors.items():
        require(anchor["receiptSha256"]=="b1df3da8f429d5fcd6e2ca2f4ac01ea8f4781aaee2e9fa88409ef1f736363ffb"
                and program not in manifest["protectedProgramAnchors"],"native metadata mechanics selection conflicts with accepted behavior")
        proof=json.loads(read_pinned(anchor["receiptPath"],anchor["receiptSha256"]))
        require(proof["schema"]=="current-minccino-mechanics-selection/1" and proof["programSha256"]==program
                and proof["selectedNativeId"]==anchor["nativeId"]==555
                and proof["selectedAttackId"]==anchor["attackId"]==792
                and proof["candidateResults"]["555"]=={"observations":16,"matched":16,"differences":0}
                and proof["preservedC9ProgramBehaviorCases"]==0
                and proof["cardSourceIDs"]==anchor["cardSourceIDs"] and proof["cardSha256"]==anchor["cardSha256"],
                "native metadata mechanics selection does not bind its measured source/physical identity")
        for name,pin in proof["outputs"].items():read_pinned(Path(anchor["receiptPath"]).parent/name,pin)
        replay_binding=anchor["candidateReplay"]
        require(replay_binding["sha256"]=="3c6aa436253d087bbd19c314ac0356c6ef5cb6fe4fa8dfef975e77bcac0c2470",
                "native metadata mechanics anchor lacks its reviewed fresh candidate replay")
        replay=json.loads(read_pinned(replay_binding["path"],replay_binding["sha256"]))
        engine={name:pin for name,pin in replay["sourceSha256"].items()
                if not name.startswith("experiments/") and Path(name).suffix in {".h",".cpp",".def",".props",".vcxproj"}}
        require(replay["schema"]=="current-standard-native-candidate-replay/1"
                and replay["sourceCommit"]==receipt["engineSourceCommit"]
                and replay["librarySha256"]==receipt["librarySha256"] and engine==receipt["engineSourceSha256"]
                and replay["selectedNativeId"]==anchor["nativeId"] and replay["candidateResults"]==proof["candidateResults"]
                and replay["actualDllCatalogEqual"] is True and replay["successful"] is True
                and replay["sourceRecaptured"] is False,"native metadata mechanics replay uses a different actual engine or source contract")
        for name,pin in replay["outputs"].items():read_pinned(Path(replay_binding["path"]).parent/name,pin)
    for previous in parent["selected"]:
        current=items.get(previous["cardSourceID"])
        require(current is not None and all(current[field]==previous[field] for field in
                ("cardSha256","programSha256","projectionSha256","rawLabelProjectionSha256")),
                "native metadata extension dropped or changed a prior source metadata observation")
        if current["cabtCardId"]!=previous["cabtCardId"]:
            anchor=anchors.get(current["programSha256"])
            require(anchor is not None and current["cabtCardId"]==anchor["nativeId"],
                    "native metadata extension silently rebound a prior physical identity")


def verify_metadata_graph_extension(manifest,items,receipt,candidate):
    extension=manifest["graphIdentityExtension"]
    require(extension["schema"]=="native-metadata-graph-identity-extension/1"
            and extension["parentManifestSha256"]==NATIVE_METADATA_C13_MANIFEST_SHA256
            and extension["canonicalReceiptSha256"]==NATIVE_METADATA_TRAINER_CANONICAL_SHA256
            and extension["graphReceiptSha256"]==NATIVE_METADATA_TRAINER_GRAPH_SHA256
            and extension["addedRows"]==99 and extension["priorRowsUnchanged"]==18957,
            "native metadata graph identity extension changed its reviewed domain")
    parent=json.loads(read_pinned(extension["parentManifestPath"],NATIVE_METADATA_C13_MANIFEST_SHA256))
    prior={row["cardSourceID"]:row for row in parent["selected"]}
    require(len(prior)==18957 and all(items.get(sid)==row for sid,row in prior.items())
            and all(manifest[key]==parent[key] for key in ("anchors","protectedProgramAnchors","mechanicsProgramAnchors","inventoryExtension")),
            "native metadata graph identity extension changed a prior source selection or anchor")
    canonical_path=Path(extension["canonicalReceiptPath"])
    canonical=json.loads(read_pinned(canonical_path,NATIVE_METADATA_TRAINER_CANONICAL_SHA256))
    require(canonical["schema"]=="standard-trainer-canonical-proposals/1"
            and canonical["counts"]=={"proposedPrograms":7,"proposedPrintings":99,"heldPrograms":11,
                                      "newFormalAdmissions":0,"acceptedIdentityConflicts":0}
            and canonical["sourceCommit"]==receipt["engineSourceCommit"]
            and canonical["actualLibrarySha256"]==receipt["librarySha256"],
            "native metadata canonical graph proposals changed actual candidate or counts")
    for name,pin in canonical["outputs"].items():read_pinned(canonical_path.parent/name,pin)
    graph_binding=canonical["measurementReceipt"]
    require(graph_binding["sha256"]==NATIVE_METADATA_TRAINER_GRAPH_SHA256,"native metadata graph measurement changed")
    graph_path=Path(graph_binding["path"])
    graph=json.loads(read_pinned(graph_path,NATIVE_METADATA_TRAINER_GRAPH_SHA256))
    require(graph["schema"]=="standard-trainer-native-identity-classes/1"
            and graph["sourceCommit"]==candidate["sourceCommit"] and graph["sourceSha256"]==candidate["sourceSha256"]
            and graph["actualLibrarySha256"]==receipt["librarySha256"] and graph["actualDllCatalogEqual"] is True
            and graph["counts"]["registeredIdentityConsumers"]==0,
            "native metadata graph fixture changed engine source or concrete identity consumers")
    for name,pin in graph["outputs"].items():read_pinned(graph_path.parent/name,pin)
    registry=json.loads(read_pinned(graph["registryReceipt"]["path"],graph["registryReceipt"]["sha256"]))
    require(registry["sourceSha256"]==candidate["sourceSha256"] and registry["sourceCommit"]==candidate["sourceCommit"]
            and registry["objectSha256"]==graph["registryObjectSha256"],"native metadata graph object closure changed")
    read_pinned(registry["objectPath"],registry["objectSha256"])
    require(canonical["sourceOrderingFileSha256"]==candidate["sourceSha256"]["All.h"],
            "native metadata canonical registration order changed")
    normalized=json.loads((graph_path.parent/"normalized-graphs.json").read_bytes())
    raw_graph=json.loads((graph_path.parent/"graph.stdout").read_bytes())
    require(not raw_graph["references"] and (raw_graph["cardCount"],raw_graph["attackCount"],raw_graph["skillCount"])==(10216,14255,2648),
            "native metadata graph measurement omitted the full registry or identity consumers")
    proposals=json.loads((canonical_path.parent/"proposals.json").read_bytes())
    anchors=extension["programAnchors"]
    require(len(proposals)==len(anchors)==7,"native metadata graph canonical program domain changed")
    added={}
    for proposal in proposals:
        program=proposal["programSha256"];anchor=anchors[program]
        ids=proposal["nativeCandidateIds"];chosen=proposal["canonicalNativeIdProposal"]
        require(proposal["allCandidateGraphsEqual"] is True and len(proposal["graphClasses"])==1
                and proposal["graphClasses"][0]["nativeIds"]==ids and chosen in ids
                and not proposal["references"] and not proposal["sourceNumericReferences"]
                and not proposal["qualifiedSelectionConflicts"] and not any(proposal["acceptedIdentityConflicts"].values())
                and proposal["canonicalPolicy"]=="prefer CardImpl.h; otherwise first actual pinned InitializeAll call; then first registration in that file"
                and chosen==min(proposal["sourceOrderingEvidence"],key=lambda row:tuple(row["rank"]))["nativeId"]
                and all(digest(normalized[str(cid)])==proposal["graphClasses"][0]["graphSha256"] for cid in ids),
                "native metadata graph canonical choice lacks full equality, identity guards or registration order")
        require(anchor=={"nativeId":chosen,"nativeCandidateIds":ids,"graphSha256":proposal["graphClasses"][0]["graphSha256"],
                         "canonicalPolicy":proposal["canonicalPolicy"],"printings":proposal["printings"]},
                "native metadata graph program anchor differs from its measured proposal")
        for printing in proposal["printings"]:
            sid=printing["cardSourceID"]
            require(sid not in added and sid not in prior and printing["programSha256"]==program
                    and all(items[sid][key]==printing[key] for key in ("cardSourceID","cardSha256","programSha256"))
                    and items[sid]["cabtCardId"]==chosen and items[sid]["nativeCandidateIds"]==ids,
                    "native metadata graph printing changed raw source or chosen native identity")
            added[sid]=program
    require(len(added)==99 and set(items)==prior.keys()|added.keys(),"native metadata graph extension added an unreviewed printing")
    for name,binding in canonical["protectedReports"].items():
        report=json.loads(read_pinned(binding["path"],binding["sha256"]))
        if name=="c13":require(report["engine"]["sourceCommit"]==candidate["sourceCommit"],"native metadata current accepted report engine changed")
        require(all(case["observedNativeId"]==anchors[case["programSha256"]]["nativeId"] for case in report["cases"]
                    if case["status"]=="oracle_matched" and case["programSha256"] in anchors),
                "native metadata graph canonical selection conflicts with accepted physical identity")


def native_metadata_domain(artifacts, candidate, source_rows):
    raw=Path(artifacts["nativeMetadataManifest"]).read_bytes()
    manifest_sha=sha256(raw).hexdigest()
    require(manifest_sha in NATIVE_METADATA_DOMAINS,"native metadata manifest changed")
    domain=NATIVE_METADATA_DOMAINS[manifest_sha]
    manifest=json.loads(raw)
    policy=manifest.get("labelPolicy")
    require(policy==(NATIVE_METADATA_LABEL_POLICY if domain["labelPolicy"] else None),"native metadata label policy differs from pinned domain")
    require(bool(manifest.get("inventoryExtension"))==domain.get("inventoryExtension",False),"native metadata inventory policy differs from pinned domain")
    require(bool(manifest.get("graphIdentityExtension"))==domain.get("graphIdentityExtension",False),"native metadata graph identity policy differs from pinned domain")
    require(manifest["schema"]=="native-full-metadata-inputs/1" and manifest["corpusSha256"]==LIVE_PINS.corpus_sha256
            and manifest["indexSha256"]==LIVE_PINS.index_sha256 and manifest["denominator"]=={"rows":22945,"programs":9001,"requiredCases":93514},"native metadata full source denominator changed")
    receipt_raw=Path(artifacts["nativeMetadataSourceReceipt"]).read_bytes();receipt=json.loads(receipt_raw)
    family=candidate.get("fixtureFamilies",{}).get("nativeMetadata",{})
    require(family.get("sourceReceiptSha256")==sha256(receipt_raw).hexdigest()
            and family.get("manifestSha256")==manifest_sha,"native metadata supplemental source receipt is not pinned by candidate")
    if manifest.get("inventoryExtension"):
        require(manifest["inventoryExtension"]["currentObservationReceiptSha256"]==sha256(receipt_raw).hexdigest(),
                "native metadata extension does not bind the exact fresh source receipt")
    require(receipt["schema"]=="supplemental-native-metadata-probe/1" and receipt["actualDllCatalogEqual"] is True
            and receipt["fixtureSha256"]==sha256(Path(artifacts["nativeMetadataFixture"]).read_bytes()).hexdigest()
            and receipt["librarySha256"]==sha256(Path(artifacts["library"]).read_bytes()).hexdigest(),"native metadata actual fixture/library changed")
    engine={p:h for p,h in candidate["sourceSha256"].items() if not p.startswith("experiments/") and Path(p).suffix in {".h",".cpp",".def",".props",".vcxproj"}}
    require(engine==receipt["engineSourceSha256"] and all(receipt["probeSourceSha256"].get(p)==h for p,h in NATIVE_METADATA_PROBE_SOURCES.items()),"native metadata compiled engine core or reviewed probe source changed")
    by_source={};program_sources=defaultdict(list)
    for card,_,row_hash,program in source_rows:
        sid=card["cardSourceID"]
        require(sid not in by_source,"native metadata duplicate source row")
        by_source[sid]=(card,row_hash,program);program_sources[program].append(sid)
    require(len(by_source)==22945 and len(program_sources)==9001,"native metadata source inventory is incomplete")
    inventory=[{"programSha256":p,"cardSourceIDs":ss,"cardSha256":{s:by_source[s][1] for s in ss}} for p,ss in sorted(program_sources.items())]
    require(digest(inventory)==manifest["sourceProgramInventorySha256"],"native metadata complete source row/program inventory changed")
    items={};projections={};program_ids=defaultdict(set)
    for item in manifest["selected"]:
        sid=item["cardSourceID"];card,row_hash,program=by_source[sid]
        require(sid not in items and row_hash==item["cardSha256"] and program==item["programSha256"]
                and manifest["programSourceIDs"][program]==program_sources[program],"native metadata exact source/printing binding changed")
        source=source_metadata(card);raw_projection=source_projection(source,source["passiveActions"])
        projection=metadata_label_projection(raw_projection,policy)
        if policy is not None:require(digest(raw_projection)==item["rawLabelProjectionSha256"],"native metadata raw English labels changed")
        require(digest(projection)==item["projectionSha256"],"native metadata raw source differs from measured typed decoding")
        items[sid]=item;projections[sid]=projection;program_ids[program].add(item["cabtCardId"])
    require(len(items)==domain["rows"] and len(program_ids)==domain["programs"]
            and all(len(ids)==1 for ids in program_ids.values()),"native metadata selected domain or complete program identity changed")
    if policy is not None:
        for program,ids in program_ids.items():
            hashes=set()
            for sid in program_sources[program]:
                source=source_metadata(by_source[sid][0])
                hashes.add(digest(metadata_label_projection(source_projection(source,source["passiveActions"]),policy)))
            require(len(hashes)==1,"native metadata label policy differs across full source printings")
            anchor=manifest["protectedProgramAnchors"].get(program)
            if anchor is not None:
                require(ids=={anchor["nativeId"]} and bool(anchor["acceptedCaseIds"])
                        and anchor["reportSha256"]=="c309cca020f229b6bba0a13307ea1b563ae4ac61d95d1441307be11bc7face6c","native metadata normalized identity conflicts with accepted physical program ID")
    return manifest,items,projections,receipt


def run_native_metadata(cases,artifacts,candidate,cards,source_rows):
    from tcg.candidate_replay import candidate_replay,replay_fixture,retained_metadata
    replay=candidate_replay(artifacts,candidate)
    require({"nativeMetadataManifest","nativeMetadataFixture","nativeMetadataSourceReceipt","library"}<=artifacts.keys(),"native metadata requires a pinned manifest, supplemental source receipt, fixture and actual DLL")
    source_artifacts=replay["parentArtifacts"] if replay else artifacts
    source_candidate=replay["parentCandidate"] if replay else candidate
    manifest,items,source_projections,receipt=native_metadata_domain(source_artifacts,source_candidate,source_rows)
    if replay:replay_fixture(replay,"nativeMetadata.fixture",artifacts["nativeMetadataFixture"])
    selected=[];seen=set()
    for case in cases:
        parts=case["caseId"].split("/")
        require(len(parts)==3 and parts[0]=="row" and parts[2]=="metadata_identity" and parts[1] in items,"native metadata cannot admit effects, rules or unmapped rows")
        item=items[parts[1]];scenario=native_metadata_scenario(item)
        require(manifest.get("labelPolicy") is not None or item["programSha256"] not in NATIVE_METADATA_HELD_PROGRAM_IDS,"native metadata candidate conflicts with an already accepted native behavior identity")
        require(case["caseId"] not in seen and case["scenario"]==scenario and type(case["cabtCardId"]) is int and case["cabtCardId"]==item["cabtCardId"],"native metadata case differs from exact measured source identity")
        selected.append((case,item,scenario));seen.add(case["caseId"])
    def execute(*argv):
        result=subprocess.run([str(artifacts["nativeMetadataFixture"]),*argv],capture_output=True,timeout=120,check=True)
        return json.loads(result.stdout)
    registry=execute("registry");library=ctypes.CDLL(str(artifacts["library"]))
    library.AllCard.argtypes=[];library.AllCard.restype=ctypes.c_char_p;library.AllAttack.argtypes=[];library.AllAttack.restype=ctypes.c_char_p
    card_bytes,attack_bytes=library.AllCard(),library.AllAttack()
    require(bool(card_bytes) and bool(attack_bytes) and json.loads(card_bytes)==list(cards.values())
            and registry=={"entries":json.loads(card_bytes),"attacks":json.loads(attack_bytes)},"native metadata actual probe catalog differs from initialized candidate DLL")
    raw=execute();skills=execute("skills")
    inventory=({"cards":10251,"attacks":14305,"skills":2648} if replay else
               NATIVE_METADATA_DOMAINS[sha256(Path(artifacts["nativeMetadataManifest"]).read_bytes()).hexdigest()]["nativeCounts"])
    require(isinstance(raw,list) and len(raw)==inventory["cards"] and {r["cardId"] for r in raw}==set(cards)
            and len({r["cardId"] for r in raw})==len(raw) and len(registry["attacks"])==inventory["attacks"]
            and len(skills)==inventory["skills"] and len({r["skillId"] for r in skills})==len(skills),"native metadata full card/attack/skill export is incomplete")
    prior=retained_metadata(replay,raw,registry,skills) if replay else None
    if manifest.get("inventoryExtension"):
        prior_inventory=NATIVE_METADATA_DOMAINS[sha256(Path(source_artifacts["nativeMetadataManifest"]).read_bytes()).hexdigest()]["nativeCounts"]
        require(manifest["nativeInventoryCounts"]==prior_inventory,"native metadata extension inventory count contract changed")
        verify_metadata_inventory_extension(manifest,receipt,items,prior["native-raw.json"] if replay else raw,
                                           prior["registry.json"] if replay else registry,
                                           prior["native-skills.json"] if replay else skills)
    if manifest.get("graphIdentityExtension"):
        verify_metadata_graph_extension(manifest,items,receipt,source_candidate)
    by_internal=internal_catalog(raw);by_signature=defaultdict(list);by_exact_signature=defaultdict(list);native={};policy=manifest.get("labelPolicy")
    for row in raw:
        try:
            raw_projection=native_projection(row,by_internal);projection=metadata_label_projection(raw_projection,policy)
            native[row["cardId"]]=projection;by_signature[digest(projection)].append(row["cardId"])
            by_exact_signature[digest(raw_projection)].append(row["cardId"])
        except HeldMetadata:pass
    for sid,item in items.items():
        ids=sorted(by_signature.get(item["projectionSha256"],[]))
        require(ids==item["nativeCandidateIds"] and item["cabtCardId"] in ids,"native metadata full candidate identity changed")
        anchor=manifest["anchors"].get(sid)
        if policy is None:
            require((anchor==item["cabtCardId"] and item["method"]=="preserved_c9_measured_identity") if anchor is not None
                    else len(ids)==1 and item["method"]=="unique_full_metadata_identity","native metadata changed prior measured ID or admitted an ambiguous new identity")
        else:
            exact_ids=sorted(by_exact_signature.get(item["rawLabelProjectionSha256"],[]))
            require(exact_ids==item["exactRawLabelNativeCandidateIds"],"native metadata raw-label exact candidates changed")
            program_anchor=manifest["protectedProgramAnchors"].get(item["programSha256"])
            if anchor is not None:chosen,method=anchor,"preserved_c9_measured_identity"
            elif program_anchor is not None:chosen,method=program_anchor["nativeId"],"preserved_c9_measured_program_identity"
            elif item["programSha256"] in manifest.get("mechanicsProgramAnchors",{}):
                mechanics=manifest["mechanicsProgramAnchors"][item["programSha256"]]
                require(sid in mechanics["cardSourceIDs"] and mechanics["cardSha256"][sid]==item["cardSha256"]
                        and mechanics["projectionSha256"]==item["projectionSha256"],"native metadata mechanics anchor changed full printing metadata")
                chosen,method=mechanics["nativeId"],"measured_source_mechanics_identity"
            elif item["programSha256"] in manifest.get("graphIdentityExtension",{}).get("programAnchors",{}):
                graph_anchor=manifest["graphIdentityExtension"]["programAnchors"][item["programSha256"]]
                chosen,method=graph_anchor["nativeId"],"measured_native_graph_canonical_identity"
            elif len(exact_ids)==1:chosen,method=exact_ids[0],"unique_exact_full_metadata_identity"
            else:
                require(len(ids)==1,"native metadata normalized labels leave an ambiguous new physical identity")
                chosen,method=ids[0],"unique_normalized_full_metadata_identity"
            require(chosen==item["cabtCardId"] and method==item["method"],"native metadata label policy changed a protected or exact physical identity")
        require(native[item["cabtCardId"]]==source_projections[sid],"actual native metadata differs from exact source metadata")
    observations=[]
    for case,item,scenario in selected:
        response={k:item[k] for k in ("cardSourceID","cardSha256","programSha256","cabtCardId")}
        response["projection"]=native[item["cabtCardId"]]
        observations.append({"caseId":case["caseId"],"cabtCardId":item["cabtCardId"],"scenarioSha256":digest(scenario),"observation":{"commands":scenario["commands"],"responses":[response]},"scenarioComponents":[{"suite":scenario["suite"],"scenarioSha256":digest(scenario)}]})
    return observations
