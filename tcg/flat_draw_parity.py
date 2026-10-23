import ctypes
from collections import defaultdict
from hashlib import sha256
import json
from pathlib import Path
import re
import subprocess

from tcg.live_parity import LIVE_PINS, digest

FLAT_DRAW_FACETS = ("effect", "legal_prompts", "information_boundaries", "interactions")
FLAT_DRAW_MANIFEST_SHA256 = "cfc30459c190b66ac17e621ae990d43af128f72364253c66fc3f2d51fdaf77f4"
REVIEWED_SOURCES = {
    "CardImpl.h": "286a51820d36f9b60b5b13c58bc2edff352eb4050581ddadf1960f13fd6f21a9",
    "GeneratedFlatDrawCards.h": "8ca4985192c9c17209dfbc71ca9a6bf2076015d23019f2fcd49d78e04147d0b8",
    "experiments/flat_draw_family/cabt.cpp": "e4ea8efb0b40df2788ccddcc08d8152c3707c432279c8f64b40e4dc42b805f3b",
    "experiments/flat_draw_family/Commands.h": "912566e34a795163dda178aa57d43c9244d8193efc296555cb4b3204a1081e94",
}
EXPANDED_REVIEWED_SOURCES = {
    "experiments/flat_draw_family/cabt.cpp": "6001b4c0d20593e8453d594b329cddd1c192031610d05de0e9322868fae6e585",
}
TYPES = {1:"Colorless",2:"Fire",3:"Grass",4:"Water",5:"Fighting",6:"Fairy",7:"Darkness",8:"Lightning",9:"Dragon",10:"Metal",11:"Psychic"}
SYMBOLS = dict(zip("CRGWFDLNMPY", ("Colorless","Fire","Grass","Water","Fighting","Darkness","Lightning","Dragon","Metal","Psychic","Fairy")))
CARD_KEYS = set("cardArchetypeID cardName cardProperties cardSourceID cardTyping evolvesFrom evolvesFromBase evolvesInto hp logicSourceID prizeValue releaseDate resistance resistanceAmt retreatCost useActions weakness weaknessAmt cardStage".split())
ACTION_KEYS = set("actionGuid actionName actionProperties actionSourceID actionType energyCost subActions".split())
MODES = ("sufficient","deck-empty","deck-short","deck-exact","none","short","wrong","first-turn","bench","asleep","paralyzed","confused-heads","confused-tails","weakness","resistance","knockout","last-prize","no-bench")
SCALARS = {"programSha256":str,"cardSourceID":str,"actionIndex":int,"mode":str,"legal":bool,"currentPlayer":int,"gameOver":bool,"actorWon":bool,"actorDamage":int,"defenderDamage":int,"actorDiscarded":bool,"defenderDiscarded":bool,"prizes":int,"enemyPrizes":int,"coins":int}
ZONES = ("hand","deck","enemyDeck","discard","enemyDiscard")

def _require(condition, reason):
    if not condition:raise ValueError(reason)

def _fields(value, allowed, context):
    _require(isinstance(value,dict) and not set(value)-allowed, "flat-draw unsupported "+context+" fields")

def _kind(name):return "MatchLogic."+name+", MatchLogic"

def _constant(value, maximum, context):
    _fields(value,{"$type","value1"},context)
    _require(value.get("$type")==_kind("SingleVariableTotal"),"flat-draw nonconstant "+context)
    number=value.get("value1",{});_fields(number,{"$type","explicitValue"},context+" number")
    count=number.get("explicitValue")
    _require(number.get("$type")==_kind("ExplicitNumberVariable") and type(count) is int and 0<count<=maximum,"flat-draw invalid "+context+" number")
    return count

def _draw(node):
    _fields(node,{"$type","editorSettings","subActionName","conditions","from"},"draw")
    _require(node.get("$type")==_kind("MoveCards") and node.get("editorSettings","DrawCards")=="DrawCards","flat-draw unsupported movement")
    source=node.get("from",{});_fields(source,{"entityLocations","groupingSize"},"draw source")
    _require(source.get("entityLocations")==[{"category":11,"entityPositions":[2]}],"flat-draw source must be own deck")
    count=_constant(source.get("groupingSize",{}),10,"draw")
    conditions=node.get("conditions",[])
    _require(isinstance(conditions,list) and len(conditions)<=1,"flat-draw unsupported effect-local conditions")
    if conditions:
        condition=conditions[0];_fields(condition,{"$type","entity","notes"},"draw condition")
        _require(condition.get("$type")==_kind("ExistenceConditional"),"flat-draw unsupported effect-local condition")
        entity=condition.get("entity",{});_fields(entity,{"entityLocations","groupingSize","editorSettings"},"condition target")
        _require(entity.get("editorSettings","SelfPlayerDeck")=="SelfPlayerDeck"
                 and entity.get("entityLocations")==[{"category":11,"entityPositions":[2]}]
                 and entity.get("groupingSize")=={"$type":_kind("AllNumberVariableTotal")},"flat-draw unsupported local deck condition")
    return count,bool(conditions)

def _damage(node, resolve):
    _fields(node,{"$type","damageValue","editorSettings","subActionName","target"},"damage")
    _fields(resolve,{"$type","subActionName"},"resolve")
    _require(node.get("$type")==_kind("ApplyDamage") and resolve.get("$type")==_kind("ResolveAttack")
             and node.get("editorSettings","DamageOpponentsActive")=="DamageOpponentsActive","flat-draw unsupported damage/resolve")
    target=node.get("target",{});_fields(target,{"entityLocations","groupingSize","editorSettings"},"damage target")
    _require(target.get("entityLocations")==[{"category":12,"entityPositions":[7]}]
             and target.get("groupingSize")=={"$type":_kind("AllNumberVariableTotal")}
             and target.get("editorSettings","OpponentActive")=="OpponentActive","flat-draw damage must target opponent Active")
    amount=_constant(node.get("damageValue",{}),1000,"damage")
    _require(amount%10==0,"flat-draw unsupported damage amount")
    return amount

def parse_flat_draw_card(card, names):
    _fields(card,CARD_KEYS,"card")
    _require(card.get("cardProperties")==[0] and card.get("prizeValue")==1,"flat-draw requires ordinary single-prize Pokemon")
    stage=card.get("cardStage",1);name=card.get("cardName")
    _require(type(stage) is int and stage in (1,2,3) and isinstance(name,str) and name
             and not any(text in name for text in ("'s ","’s ","Team ","[","]")),"flat-draw unsupported stage or owner/special name")
    internal=names.get(name,set());_require(len(internal)<=1,"flat-draw ambiguous internal name")
    predecessor=None;predecessor_internal=None
    if stage==1:
        _require("evolvesFrom" not in card and card.get("evolvesFromBase")==[name],"flat-draw Basic evolution metadata differs")
    else:
        predecessor=card.get("evolvesFrom");bases=card.get("evolvesFromBase")
        _require(isinstance(predecessor,str) and len(names.get(predecessor,set()))==1
                 and isinstance(bases,list) and len(bases)==1 and len(names.get(bases[0],set()))==1
                 and (stage!=2 or bases==[predecessor]),"flat-draw unresolved evolution closure")
        predecessor_internal=next(iter(names[predecessor]))
    typing=card.get("cardTyping",[]);hp=card.get("hp");retreat=card.get("retreatCost",0)
    _require(isinstance(typing,list) and len(typing)==1 and typing[0] in TYPES
             and type(hp) is int and 0<hp<=1000 and type(retreat) is int and 0<=retreat<=4,"flat-draw unsupported Pokemon metadata")
    defenses={}
    for field,amount in (("weakness",2),("resistance",30)):
        values=card.get(field,[])
        _require(isinstance(values,list) and len(values)<=1 and all(v in TYPES and v!=1 for v in values)
                 and (not values or card.get(field+"Amt")==amount)
                 and (values or card.get(field+"Amt",0)==0),"flat-draw unsupported "+field)
        defenses[field]=TYPES[values[0]] if values else None
    actions=card.get("useActions",[]);_require(isinstance(actions,list) and 1<=len(actions)<=2,"flat-draw unsupported action count")
    attacks=[]
    for action in actions:
        _fields(action,ACTION_KEYS,"action")
        _require(action.get("actionType")==1 and action.get("actionProperties")==[1]
                 and isinstance(action.get("actionName"),str) and action["actionName"]
                 and isinstance(action.get("actionGuid"),str),"flat-draw requires exact plain attacks")
        cost=action.get("energyCost","")
        _require(isinstance(cost,str) and re.fullmatch(r"(?:\{[CRGWFDLMNPY]\}){0,8}",cost) is not None,"flat-draw unsupported typed energy cost")
        subs=action.get("subActions",[]);_require(isinstance(subs,list),"flat-draw invalid subactions")
        types=[s.get("$type") for s in subs];draw=0;local=False;draw_only=False
        if types==[_kind("MoveCards")]:
            draw,local=_draw(subs[0]);damage=0;draw_only=True
        elif types==[_kind("ApplyDamage"),_kind("ResolveAttack"),_kind("MoveCards")]:
            damage=_damage(subs[0],subs[1]);draw,local=_draw(subs[2])
        elif types==[_kind("ApplyDamage"),_kind("ResolveAttack")]:damage=_damage(subs[0],subs[1])
        else:raise ValueError("flat-draw unsupported exact action graph")
        attacks.append({"name":action["actionName"],"damage":damage,"energy":[SYMBOLS[s] for s in re.findall(r"\{(.)\}",cost)],"source_guid":action["actionGuid"],"draw":draw,"local_deck_exists":local,"draw_only":draw_only})
    _require(any(a["draw"] for a in attacks),"flat-draw requires a fixed-count draw attack")
    return {"name":name,"internal_name":next(iter(internal),name),"hp":hp,"retreat":retreat,"typing":TYPES[typing[0]],**defenses,"attacks":attacks,"stage":stage,"predecessor":predecessor,"predecessor_internal":predecessor_internal}

def flat_draw_commands(item, action_index=None):
    if action_index is not None and (type(action_index) is not int or not 0<=action_index<len(item["card"]["attacks"])):
        raise ValueError("flat-draw action index differs from the printed source")
    commands=[]
    for index,attack in enumerate(item["card"]["attacks"]):
        if action_index is not None and index!=action_index:continue
        for mode in MODES:
            if mode in ("weakness","resistance") and item["card"]["typing"]=="Colorless":continue
            deck=0 if mode=="deck-empty" else max(0,attack["draw"]-1) if mode=="deck-short" else attack["draw"] if mode=="deck-exact" else max(4,attack["draw"]+2)
            commands.append({"programSha256":item["programSha256"],"cardSourceID":item["cardSourceIDs"][0],"cabtCardId":item["cabtCardId"],"actionIndex":index,"nativeActionId":item["attackIds"][index],"mode":mode,"deck":deck,"prizes":1 if mode=="last-prize" else 3,"seed":17})
    return commands

def flat_draw_metadata_scenario(item, source_id):
    return {"suite":"flat-draw-metadata/1","commands":[{"operation":"nativeFlatDrawMetadata","programSha256":item["programSha256"],"cardSourceID":source_id,"cardSha256":item["cardSha256"][source_id]}]}

def flat_draw_scenario(item, channel, action_index, facet):
    if channel!="useActions" or facet not in FLAT_DRAW_FACETS:raise ValueError("flat-draw admits only measured printed attack facets")
    return {"suite":f"flat-draw-{channel}-{action_index}-{facet}/1","commands":flat_draw_commands(item,action_index)}

def _source_programs(artifacts, candidate, cards, source_rows):
    raw=artifacts["flatDrawManifest"].read_bytes()
    _require(sha256(raw).hexdigest()==FLAT_DRAW_MANIFEST_SHA256,"flat-draw manifest changed")
    _require(all(candidate["sourceSha256"].get(p) in {h,EXPANDED_REVIEWED_SOURCES.get(p,h)} for p,h in REVIEWED_SOURCES.items()),"flat-draw reviewed generated header, fixture or command source changed")
    manifest=json.loads(raw)
    _require(manifest["schema"]=="flat-draw-family/1" and manifest["corpusSha256"]==LIVE_PINS.corpus_sha256
             and manifest["denominator"]=={"programs":9001,"rows":22945,"requiredCases":93514},"flat-draw source domain changed")
    catalog_raw=Path(manifest["catalogPath"]).read_bytes()
    _require(sha256(catalog_raw).hexdigest()==manifest["catalogSha256"],"flat-draw canonical native name catalog changed")
    catalog=json.loads(catalog_raw);names=defaultdict(set)
    for row in catalog["names"]:names[row["name"]].add(row["internalName"])
    by_source={};program_rows=defaultdict(list)
    for card,_,row_hash,program in source_rows:
        sid=card["cardSourceID"]
        _require(sid not in by_source,"flat-draw duplicate source identity")
        by_source[sid]=(card,row_hash,program);program_rows[program].append(sid)
    _require(len(by_source)==22945 and len(program_rows)==9001,"flat-draw full sorted-ordinal denominator changed")
    recognized={};ordinals={}
    for ordinal,(program,sources) in enumerate(sorted(program_rows.items())):
        ordinals[program]=ordinal
        try:recognized[program]=parse_flat_draw_card(by_source[sources[0]][0],names)
        except ValueError:pass
    _require(len(recognized)==78,"flat-draw strict complete family recognition changed")
    items={};row_items={};used_attacks={a["attackId"] for a in catalog["attacks"]};next_attack=10000
    for item in manifest["selected"]:
        program=item["programSha256"];sources=item["cardSourceIDs"]
        _require(program not in items and program in recognized and sources==program_rows[program]
                 and set(item["cardSha256"])==set(sources) and item["card"]==recognized[program]
                 and type(item["cabtCardId"]) is int and item["cabtCardId"]==2000000+ordinals[program]
                 and item["cabtCardId"] in cards,"flat-draw exact program, printings, metadata or sorted-ordinal native ID changed")
        expected=[]
        for _ in item["card"]["attacks"]:
            while next_attack in used_attacks:next_attack+=1
            expected.append(next_attack);used_attacks.add(next_attack);next_attack+=1
        _require(item["attackIds"]==expected,"flat-draw deterministic unused attack IDs changed")
        for sid in sources:
            card,row_hash,source_program=by_source[sid]
            _require(sid not in row_items and source_program==program and row_hash==item["cardSha256"][sid]
                     and digest({k:v for k,v in card.items() if k not in {"cardSourceID","releaseDate"}})==program
                     and parse_flat_draw_card(card,names)==item["card"],"flat-draw raw printing or complete source program changed")
            row_items[sid]=item
        domain=item["card"]
        if domain["stage"]>1:
            _require(any(c["cardType"]==0 and c["name"]==domain["predecessor"] and c["evolutionType"]==domain["stage"]-1 for c in cards.values())
                     and cards[item["cabtCardId"]]["evolvesFrom"]==domain["predecessor"],"flat-draw current actual catalog lacks the exact staged parent closure")
        items[program]=item
    _require(set(items)==set(recognized) and list(items)==sorted(items) and len(row_items)==181
             and sum(len(i["card"]["attacks"]) for i in items.values())==145
             and manifest["commands"]==[c for i in items.values() for c in flat_draw_commands(i)]
             and len(manifest["commands"])==2530,"flat-draw complete78/181/145/2530 inventory changed")
    return items,row_items

def _known(value):return isinstance(value,list) and all(isinstance(c,dict) and type(c.get("id")) is int and c["id"]>0 for c in value)

def _views(owner, opponent):
    _require(all(isinstance(v,dict) and isinstance(v.get("players"),list) and len(v["players"])==2 for v in (owner,opponent)),"flat-draw omitted actual private seat views")
    for seat in range(2):
        visible,hidden=(owner,opponent) if seat==0 else (opponent,owner);one,other=visible["players"][seat],hidden["players"][seat]
        _require(_known(one.get("hand")) and other.get("hand") is None and one.get("handCount")==len(one["hand"])
                 and one["handCount"]==other.get("handCount") and type(one.get("deckCount")) is int and one["deckCount"]==other.get("deckCount")
                 and "deck" not in one and "deck" not in other and all(_known(one.get(z)) and one[z]==other.get(z) for z in ("active","bench","discard")),"flat-draw actual private or public zone identities changed")

def flat_draw_response(row, command, item):
    _require(isinstance(row,dict) and set(row)==set(SCALARS)|set(ZONES)|{"trace","ownerView","opponentView"}
             and all(type(row[k]) is t for k,t in SCALARS.items())
             and all(isinstance(row[z],list) and all(type(v) is str for v in row[z]) for z in ZONES)
             and isinstance(row["trace"],list),"flat-draw native response shape changed")
    _require(all(row[k]==command[k] for k in ("programSha256","cardSourceID","actionIndex","mode")),"flat-draw response identity changed")
    _views(row["ownerView"],row["opponentView"])
    for step in row["trace"]:
        _require(isinstance(step,dict) and set(step)=={"prompt","selected","valid","ownerView","opponentView"}
                 and step["valid"] is True and isinstance(step["selected"],list),"flat-draw omitted actual typed native prompt answers")
        prompt=step["prompt"];selected=step["selected"]
        _require(isinstance(prompt,dict) and type(prompt.get("minCount")) is int and type(prompt.get("maxCount")) is int
                 and prompt["minCount"]<=len(selected)<=prompt["maxCount"]
                 and len(set(selected))==len(selected) and all(type(i) is int and 0<=i<len(prompt.get("option",[])) for i in selected),"flat-draw native answer violates actual declared selection constraints")
        _views(step["ownerView"],step["opponentView"])
    attack=item["card"]["attacks"][command["actionIndex"]];mode=command["mode"];cost=attack["energy"]
    legal=mode not in ("first-turn","bench","asleep","paralyzed") and not (mode in ("none","short") and cost) and not (mode=="wrong" and any(e!="Colorless" for e in cost))
    draw=min(command["deck"],attack["draw"]) if legal and mode!="confused-tails" else 0
    _require(row["legal"]==legal and row["deck"]==[f"deck{i}" for i in range(draw,command["deck"])]
             and all(f"deck{i}" in row["hand"] for i in range(draw))
             and row["coins"]==(1 if legal and mode.startswith("confused") else 0),"flat-draw actual legality, fixed draw count/order or controlled chance changed")
    owner,opponent=row["ownerView"],row["opponentView"];one,hidden=owner["players"][0],opponent["players"][0]
    _require(len(one["hand"])==len(row["hand"]) and one["deckCount"]==len(row["deck"]) and len(one["discard"])==len(row["discard"]),"flat-draw final native view disagrees with card movement")
    privacy={"ownerHandKnown":_known(one["hand"]),"opponentHandHidden":hidden["hand"] is None,"deckHiddenFromBoth":all("deck" not in p for p in owner["players"]+opponent["players"]),"activePublic":bool(one["active"]) and bool(hidden["active"]) and _known(hidden["active"]),"discardPublic":all(_known(p["discard"]) for p in owner["players"]+opponent["players"])}
    _require(all(privacy.values()),"flat-draw native privacy boundary changed")
    return {**{k:row[k] for k in (*SCALARS,*ZONES)},"privacy":privacy}

def flat_draw_metadata(item, source_id, card, attacks):
    domain=item["card"]
    _require(card["cardId"]==item["cabtCardId"] and card["name"]==domain["name"] and card["cardType"]==0
             and not card["skills"] and card["attacks"]==item["attackIds"] and card["pokemonType"]==1,"flat-draw native Card/Attack identity changed")
    types={0:1,1:3,2:2,3:4,4:8,5:11,6:5,7:7,8:10,9:9,12:6};symbols={0:"C",1:"G",2:"R",3:"W",4:"L",5:"P",6:"F",7:"D",8:"M",9:"N",12:"Y"}
    return {"programSha256":item["programSha256"],"cardSourceID":source_id,"name":card["name"],"hp":card["hp"],"retreat":card["retreatCost"],"typing":[types[card["energyType"]]],"stage":1 if card["basic"] else 2 if card["stage1"] else 3,"prizeValue":1 if card["pokemonType"]==1 else 2,"properties":[0] if card["pokemonType"]==1 else [-1],"weakness":[types[card["weakness"]]] if card["weakness"] is not None else [],"resistance":[types[card["resistance"]]] if card["resistance"] is not None else [],"weaknessAmount":2 if card["weakness"] is not None else 0,"resistanceAmount":30 if card["resistance"] is not None else 0,"attacks":[{"name":attacks[a]["name"],"energy":"".join("{"+symbols[e]+"}" for e in attacks[a]["energies"])} for a in card["attacks"]]}

def run_flat_draw_family(cases, artifacts, candidate, cards, source_rows):
    _require({"flatDrawManifest","flatDrawFixture","library"}<=artifacts.keys(),"flat-draw requires a pinned manifest, native fixture and actual DLL")
    items,row_items=_source_programs(artifacts,candidate,cards,source_rows);selected_cases=[];seen=set()
    for case in cases:
        parts=case["caseId"].split("/")
        if len(parts)==3 and parts[0]=="row" and parts[2]=="metadata_identity" and parts[1] in row_items:
            item=row_items[parts[1]];scenario=flat_draw_metadata_scenario(item,parts[1])
        elif len(parts)==5 and parts[0]=="program" and parts[1] in items and parts[2]=="useActions" and parts[3] in ("0","1") and parts[4] in FLAT_DRAW_FACETS:
            item=items[parts[1]];scenario=flat_draw_scenario(item,parts[2],int(parts[3]),parts[4])
        else:raise ValueError("flat-draw cannot admit rules, abilities, other channels or unsupported cases")
        _require(case["caseId"] not in seen and case["scenario"]==scenario and type(case["cabtCardId"]) is int and case["cabtCardId"]==item["cabtCardId"],"flat-draw case differs from its complete measured source scenario")
        seen.add(case["caseId"]);selected_cases.append((case,item,scenario,parts))
    def execute(*args):return json.loads(subprocess.run([str(artifacts["flatDrawFixture"]),*args],capture_output=True,text=True,encoding="utf-8",timeout=60,check=True).stdout)
    registry=execute("registry");library=ctypes.CDLL(str(artifacts["library"]))
    library.AllCard.argtypes,library.AllCard.restype=[],ctypes.c_char_p;library.AllAttack.argtypes,library.AllAttack.restype=[],ctypes.c_char_p
    cards_raw,attacks_raw=library.AllCard(),library.AllAttack()
    _require(bool(cards_raw) and bool(attacks_raw) and json.loads(cards_raw)==list(cards.values())
             and registry=={"entries":json.loads(cards_raw),"attacks":json.loads(attacks_raw)},"flat-draw fixture catalog differs from the actual initialized current candidate DLL")
    attacks={a["attackId"]:a for a in registry["attacks"]}
    metadata={sid:flat_draw_metadata(item,sid,cards[item["cabtCardId"]],attacks) for sid,item in row_items.items()}
    commands={(c["programSha256"],c["cardSourceID"],c["actionIndex"],c["mode"]):c for item in items.values() for c in flat_draw_commands(item)}
    raw=execute();_require(isinstance(raw,list),"flat-draw fixture omitted actual responses");responses={}
    for row in raw:
        key=(row["programSha256"],row["cardSourceID"],row["actionIndex"],row["mode"])
        _require(key in commands and key not in responses,"flat-draw native duplicate or foreign response identity")
        responses[key]=flat_draw_response(row,commands[key],items[key[0]])
    _require(responses.keys()==commands.keys() and len(responses)==2530 and len(metadata)==181,"flat-draw actual native response inventory changed")
    observations=[]
    for case,item,scenario,parts in selected_cases:
        rows=([metadata[parts[1]]] if parts[0]=="row" else [responses[(c["programSha256"],c["cardSourceID"],c["actionIndex"],c["mode"])] for c in scenario["commands"]])
        observations.append({"caseId":case["caseId"],"cabtCardId":item["cabtCardId"],"scenarioSha256":digest(scenario),"observation":{"commands":scenario["commands"],"responses":rows},"scenarioComponents":[{"suite":scenario["suite"],"scenarioSha256":digest(scenario)}]})
    return observations
