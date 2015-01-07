(* ::Package:: *)

BeginPackage["CaffeLink`"]

newNet::usage = ""
newSolver::usage = ""
solverAddNetParam::usage = ""

initCaffeLinkModule::usage = "initCaffeLinkModule[path] initializes module from caffe.proto file."
NewNetOld::usage = "NewNetOld[name] returns new net with given name."
NewSolverOld::usage = "NewSolverOld[type] returns new solver of given type."
AddLayer::usage = "AddLayer[net, type, name] appends new layer of giventype and name to the given net."
SetNetParam::usage = "SetNetParam[net, name, value] sets parameter 'name' to 'value' in the given net."
SetLayerParam::usage = "SetLayerParam[net, i, name, value] sets parameter 'name' to 'value' in layer 'i' in net, layer 'i' is ith added layer to the net."
SetSolverParam::usage = "SetSolverParam[solver, name, value] sets parameter 'name' to 'value' in the given solver."
getParamString::usage = "getParamString[net] generates protobuffer string from net compatible with Caffe."


cblasTest::usage = "Test cblas stability."

initCaffeLink::usage = "initCaffeLink[useDouble, GPUmode, devID] initializes CaffeLink library with data type double or float, sets GPU or CPU mode and device ID for GPU mode."
prepareNetFile::usage = "prepareNetFile[path] sets up a new net from protobuffer file on given path."
prepareNetString::usage = "prepareNetString[string] sets up a new net from given protobuffer string."
loadNet::usage = "loadNet[path] loads previously exported or snapshoted net after the same net was prepared using prepareNet*."
exportNet::usage = "exportNet[path] exports previously loaded net to selected path."

testNet::usage = "testNet[] runs loaded net in test mode."
trainNetString::usage = "trainNetString[solverParam] trains new net with solver protobuffer parameters given in string."
trainNetFile::usage = "trainNetFile[path] trains new net with solver protobuffer parameters taken from file."
trainNetSnapshotString::usage = "trainNetSnapshotString[solverParam, pathState] continues training from solver state file on 'pathState'. Solver parameters given as string."
trainNetSnapshotFile::usage = "trainNetSnapshotFile[path, pathState] continues training from solver state file on 'pathState'. Solver parameters given as 'path' to file."
trainNetWeightsString::usage = "trainNetWeightsString[solverParam, pathWeights] finetunes net on path 'pathWeights'. Solver parameters given as string."
trainNetWeightsFile::usage = "trainNetWeightsFile[path, pathWeights] finetunes net on path 'pathWeights'. Solver parameters given as 'path' to file."

printNetInfo::usage = "printNetInfo[] prints (in console, stdout) info about prepared net: structure, layers, sizes and counts of blobs, etc."
printWorkingPath::usage = "printWorkingPath[] prints (in console, stdout) root directory, executes 'pwd' command."

getLayerNum::usage = "getLayerNum[] returns number of layers in prepared net."
getTopBlobSize::usage = "getTopBlobSize[layerIdx] returns array of top blob dimensions: {num, channels, height, width}, 4 elements for every blob."
getBottomBlobSize::usage = "getBottomBlobSize[layerIdx] returns array of bottom blob dimensions: {num, channels, height, width}, 4 elements for every blob."
getParamBlobSize::usage = "getParamBlobSize[layerIdx] returns array of parameter blob dimensions: {num, channels, height, width}, 4 elements for every blob."

getTopBlob::usage = "getTopBlob[layerIdx, blobIdx] returns data stored in top 'blobIdx'-th blob of 'layerIdx'-th layer."
getBottomBlob::usage = "getBottomBlob[layerIdx, blobIdx] returns data stored in bottom 'blobIdx'-th blob of 'layerIdx'-th layer."
getParamBlob::usage = "getParamBlob[layerIdx, blobIdx] returns data stored in parameter 'blobIdx'-th blob of 'layerIdx'-th layer."

setTopBlob::usage = "setTopBlob[data, layerIdx, blobIdx] inserts 'data' to top 'blobIdx'-th blob of 'layerIdx'-th layer."
setBottomBlob::usage = "setBottomBlob[data, layerIdx, blobIdx] inserts 'data' to bottom 'blobIdx'-th blob of 'layerIdx'-th layer."
setParamBlob::usage = "setParamBlob[data, layerIdx, blobIdx] inserts 'data' to parameter 'blobIdx'-th blob of 'layerIdx'-th layer."
setInput::usage = "setInput[data] inserts 'data' to input blob. Work only with net without input (data) layer."
(* -------------------------------------------------------------------------- *)


Begin["`Private`"]

regexSelectMsg = RegularExpression["message .*\{(([^\{\}]\n*)*(\{([^\}]\n*)*\})?)*\}"];
regexSelectEnum = RegularExpression["enum .* \{([^\}]*\n*)*\}"];
(* basic types and enums *)
basicTypes = {"bool", "bytes", "int", "long", "int32", "int64", "uint32",
"uint64", "float", "double", "string"};
strBasicType = "*b*";
strMsgType = "*s*";
globalEnums = {};
enumsInMsg = {};
paramLists = {};

layerParTypeBased = {}; (* list of parameters based on layer type *)
(* which par. types belong to which layer type *)
layerParTypeLUT = {
{"ABSVAL","",{""}},
{"ACCURACY","",{"AccuracyParameter"}},
{"ARGMAX","",{"ArgMaxParameter"}},
{"BNLL","",{""}},
{"CONCAT","",{"ConcatParameter"}},
{"CONTRASTIVE_LOSS","",{"ContrastiveLossParameter"}},
{"CONVOLUTION","",{"ConvolutionParameter"}},
{"DATA","",{"DataParameter","TransformationParameter"}},
{"DROPOUT","",{"DropoutParameter"}},
{"DUMMY_DATA","",{"DummyDataParameter","TransformationParameter"}},
{"EUCLIDEAN_LOSS","",{""}},
{"ELTWISE","",{"EltwiseParameter"}},
{"FLATTEN","",{""}},
{"HDF5_DATA","",{"HDF5DataParameter","TransformationParameter"}},
{"HDF5_OUTPUT","",{"HDF5OutputParameter"}},
{"HINGE_LOSS","",{"HingeLossParameter"}},
{"IM2COL","",{""}},
{"IMAGE_DATA","",{"ImageDataParameter","TransformationParameter"}},
{"INFOGAIN_LOSS","",{"InfogainLossParameter"}},
{"INNER_PRODUCT","",{"InnerProductParameter"}},
{"LRN","",{"LRNParameter"}},
{"MEMORY_DATA","",{"MemoryDataParameter","TransformationParameter"}},
{"MULTINOMIAL_LOGISTIC_LOSS","",{""}},
{"MVN","",{"MVNParameter"}},
{"POOLING","",{"PoolingParameter"}},
{"POWER","",{"PowerParameter"}},
{"RELU","",{"ReLUParameter"}},
{"SIGMOID","",{"SigmoidParameter"}},
{"SIGMOID_CROSS_ENTROPY_LOSS","",{""}},
{"SILENCE","",{""}},
{"SOFTMAX","",{"SoftmaxParameter"}},
{"SOFTMAX_LOSS","",{"SoftmaxParameter"}},
{"SPLIT","",{""}},
{"SLICE","",{"SliceParameter"}},
{"TANH","",{"TanHParameter"}},
{"WINDOW_DATA","",{"WindowDataParameter","TransformationParameter"}},
{"THRESHOLD","",{"ThresholdParameter"}}}
(* -------------------------------------------------------------------------- *)


initCaffeLinkModule[caffeProto_] := Module[{clProto, msgWithEnums, globalEnums},
caffeProtoPath = Replace["CaffeProtoPath",caffeProto];
If[!FileExistsQ[caffeProtoPath],
Throw[StringJoin["file not found: ", caffeProtoPath]];
];

clProto = CleanProto[caffeProtoPath];
msgWithEnums = ParseEnumsInMsg[clProto];
globalEnums = ParseGlobalEnums[clProto];
ParseParamList[globalEnums, msgWithEnums, clProto];
];
(* -------------------------------------------------------------------------- *)


newNet[netp_] := Module[{nn, netname, i, j, lap, lname, ltype},
np = netp[[1;;-2]];
netname = Replace["name",np];

nn = NewNetOld[netname];
For[i = 1, i <= Length[np], i++,
nn = SetNetParam[nn, np[[i, 1]], np[[i, 2]]];
];

For[i = 1, i <= Length[netp[[-1]]], i++,
lap = netp[[-1, i]];
lname = Replace["name",lap];
ltype = Replace["type",lap];
nn = AddLayer[nn,ltype,lname];
For[j = 1, j <= Length[lap], j++,
nn = SetLayerParam[nn,i, lap[[j,1]],lap[[j,2]]];
];
];
nn
];
(* -------------------------------------------------------------------------- *)


NewNetOld[name_] := Module[{nn},
nn = Select[paramLists, StringMatchQ[#[[1]], "NetParameter"] &][[1]];
nn = SetOneParam[nn, "name", name];
{nn, {}}
];
(* -------------------------------------------------------------------------- *)


newSolver[solp_] := Module[{type, i},
type = Replace["solver_type",solp];

sol = NewSolverOld[type];
For[i = 1, i <= Length[solp], i++,
sol = SetSolverParam[sol, solp[[i, 1]], solp[[i, 2]]];
];
sol
];
(* -------------------------------------------------------------------------- *)


solverAddNetParam[a_, b_] := Module[{type, i},
solver = Replace["solver", {a,b}];
netp = Replace["net_param", {a,b}];

StringJoin["net_param: {\n", netp, "}\n", solver]
];


NewSolverOld[type_] := Module[{ns},
ns = Select[paramLists, StringMatchQ[#[[1]], "SolverParameter"] &][[1]];
ns = SetOneParam[ns, "solver_type", type];
{ns,{}}
];
(* -------------------------------------------------------------------------- *)


AddLayer[net_, type_, name_] := Module[{cnet, netLayers, lp, laPars, laTypPar, i},
(* create new layer parameter list and set name and type *)
layer = Select[paramLists, StringMatchQ[#[[1]], "LayerParameter"] &][[1]];
layer = SetOneParam[layer, "name", name];
layer = SetOneParam[layer, "type", type];
laPars = layer[[-1]];

(* find specific parameter for this layer type *)
laTypPar = Select[layerParTypeLUT, StringMatchQ[#[[1]], type] &];
If[Length[laTypPar] < 1, Throw["Error: unknown layer type"];];
(* select its parameter list *)
laTypPar = laTypPar[[;;,3]];
laTypPar = Select[paramLists, StringMatchQ[#[[1]], laTypPar] &];

For[i=1,i<=Length[laTypPar],i++,
  laTypPar[[i,1]] = layerParTypeBased[[
    Position[layerParTypeBased,laTypPar[[i,1]]][[1,1]],
    1]];
];

(* and add it to layer *)
laPars = Join[laPars, laTypPar];

layer[[-1]] = laPars;
netLayers = net[[2]];
AppendTo[netLayers, layer];
cnet = net;
cnet[[2]] = netLayers;
cnet
];
(* -------------------------------------------------------------------------- *)


SetNetParam[net_, name_, value_] := Module[{newParList},
newParList = SetParamRekur[net[[1]], name, value];
If[newParList != {False},
	Return[{newParList, net[[2]]}];
];

Throw[StringJoin["unknown parameter name: ", name]];
];
(* -------------------------------------------------------------------------- *)


SetLayerParam[net_, layerInx_, name_, value_] := Module[{newParList, newLayers},
newParList = SetParamRekur[net[[2,layerInx]], name, value];
If[newParList != {False},
	newLayers = net[[2]];
	newLayers[[layerInx]] = newParList;
	Return[{net[[1]], newLayers}];
];

Throw[StringJoin["unknown parameter name: ", name]];
];
(* -------------------------------------------------------------------------- *)


SetSolverParam[solver_, name_, value_] := Module[{newParList},
newParList = SetParamRekur[solver[[1]], name, value];
If[newParList != {False},
	Return[{newParList, solver[[2]]}];
];

Throw[StringJoin["unknown parameter name: ", name]];
];
(* -------------------------------------------------------------------------- *)


(* Messages can be nested in protobuffers. This goes through them recursively
until proper parameter name is found. Some timmes it is good idea provide
name as 'msgname.paramname' *)
SetParamRekur[pList_, name_, value_] := Module[{newParList, list, i},
(* try set basic params - params with value *)
newParList = SetOneParam[pList, name, value];
If[newParList != {False},
	Return[newParList];
];

(* recursively go throught all special parameters - msg *)
list = pList[[-1]];
For[i = 1, i <= Length[list], i++,
	newParList = SetParamRekur[list[[i]], name, value];
	If[newParList != {False},
		list[[i]] = newParList;
		newParList = pList;
		newParList[[-1]] = list;
		Return[newParList];
	];
];

Return[{False}];
];
(* -------------------------------------------------------------------------- *)


SetOneParam[pList_, name_, value_] := Module[{list, splName, preName, ppos},
splName = name;
(* handle compound param. names *)
If[Length[StringSplit[name,"."]] > 1,
	preName = StringSplit[name,"."][[1]];
	splName = StringSplit[name,"."][[2]];
	If[preName != pList[[1]],
    Return[{False}];
  ];
];

(* list[[2;;-2]]: 1. element is msg name or type, last element is list
of unfolded parameters with special type *)
ppos = Select[pList[[2;;-2]], StringMatchQ[#[[1]], splName] &];
If[Length[ppos] == 0,
  Return[{False}];
];
ppos = Position[pList, ppos[[1]]];

list = pList;
list[[ppos[[1]], 3]] = value;
list
];
(* -------------------------------------------------------------------------- *)


getParamString[net_] := Module[{str, li},

(* start with net params *)
str = GetParamStrRekur[net[[1]], ""];

(* layers *)
For[li = 1, li <= Length[net[[2]]], li++,
	str = StringJoin[str, "layers {\n"];
	str = StringJoin[str, GetParamStrRekur[net[[2, li]], "  "]];
	str = StringJoin[str, "}\n"];
];

Return[str]
];
(* -------------------------------------------------------------------------- *)


(* Recursively generrates protobuffer string from pList - msg.
Space is size of indentation. *)
GetParamStrRekur[pList_, space_] := Module[{str, subStr, i},
(* basic params *)
str = GetParamString[pList, space];

(* parameters with special type *)
For[i = 1, i <= Length[pList[[-1]]], i++,
	subStr = GetParamStrRekur[pList[[-1, i]], StringJoin[space, "  "]];
	If[StringLength[subStr] > 0,
		str = StringJoin[str, space, pList[[-1, i, 1]], " {\n"];
		str = StringJoin[str, subStr];
		str = StringJoin[str, space, "}\n"];
	];
];
str
];
(* -------------------------------------------------------------------------- *)


(* Generrates protobuffer string from pList - msg.
Space is size of indentation. *)
GetParamString[pList_, space_] := Module[{str, value, orig, i},
str = "";
For[i = 2, i <= Length[pList] - 1, i++,
	value = Flatten[{pList[[i, 3]]}];
	orig = ToString[pList[[i, 3]]];
	If[IsBasicOrEnum[pList[[i, 2]]] && orig != strBasicType && orig != "",
		For[j = 1, j <= Length[value], j++,
			str = StringJoin[str, space, pList[[i, 1]], ": "];
			If[pList[[i, 2]] == "string",
				str = StringJoin[str, "\"", value[[j]], "\"\n"]
				,
				str = StringJoin[str, ToString[value[[j]]], "\n"];
			];
		];
	];
];
str
];
(* -------------------------------------------------------------------------- *)


(* Checks if parametr's type can hold a value directly.
Checks if parameter's type is basic or global enum or enum in msg.*)
IsBasicOrEnum[type_] := Block[{},
res = Select[basicTypes, StringMatchQ[#, type] &];
If[Length[res] > 0, Return[True];];
res = Select[globalEnums, StringMatchQ[#, type] &];
If[Length[res] > 0, Return[True]];
res = Select[enumsInMsg, StringMatchQ[#, type] &];
If[Length[res] > 0, Return[True]];
Return[False];
];
(* -------------------------------------------------------------------------- *)


(* Cleans protobuffer file in given path. CleanProto[filePath] returns
content of file without comments, empty lines, etc. *)
CleanProto[caffeProtoPath_] := Module[{str},
str = ReadString[caffeProtoPath];
(*Remove comments, empty lines and trim spaces.*)
str = StringTrim[StringSplit[str, "\n"]];
str = StringTrim[str, RegularExpression["//.*"]];
str = Select[str, #!=""&];
StringJoin[Riffle[str, ConstantArray["\n", Length[str]]]]
];
(* -------------------------------------------------------------------------- *)


(* Parses and returns enum names from cleaned proto. *)
ParseEnumNames[clProto_] := Module[{enms},
enms = StringCases[clProto, regexSelectEnum];
enms = StringSplit[enms, " "];
(* names only *)
enms[[;;, 2]]
];
(* -------------------------------------------------------------------------- *)


(* For every msg parses its enums. Returns list:
{{msg_01, {enum01, enum02, ...}}, ...} *)
ParseEnumsInMsg[clProto_] := Module[{i, msgs, msgNames, msgWithEnums},
msgs = StringCases[clProto,regexSelectMsg];
msgNames = StringSplit[msgs, " "][[;;, 2]];

(*For every msg parse its enums.*)
enumsInMsg = {}; (* only enum names *)
msgWithEnums = ConstantArray[{},Length[msgs]]; (* msg and its own enums *)
For[i = 1, i <= Length[msgs], i++,
	AppendTo[enumsInMsg, ParseEnumNames[msgs[[i]]]];
	msgWithEnums[[i]] = {msgNames[[i]], ParseEnumNames[msgs[[i]]]};
];
enumsInMsg = Flatten[enumsInMsg];
msgWithEnums
];
(* -------------------------------------------------------------------------- *)


(* Parses and returns list of enums outside messages
in clProto - cleaned proto file. *)
ParseGlobalEnums[clProto_] := Module[{str},
str = StringReplace[clProto,regexSelectMsg -> ""];
globalEnums = ParseEnumNames[str]
];
(* -------------------------------------------------------------------------- *)


(* Creates list of lists of parameters from messages in 'clProto'
and enum lists. *)
ParseParamList[globalEnum_, msgEnum_, clProto_] := Module[
{i, msgs, parList, pars, parListUnfold, parListUnfold2},
msgs = StringReplace[clProto, regexSelectEnum-> ""];
msgs = StringCases[msgs,regexSelectMsg];
msgs = StringReplace[msgs, RegularExpression["(.*\{\n?)|(\}.*\n?)|( *\n *)"]-> ""];
msgs = StringSplit[msgs, ";"]; (* list of messages *)

parList = {};
(* convert each msg to list of parameters *)
For[i = 1, i <= Length[msgs], i++, 
	pars = StringSplit[msgs[[i]], " "][[;;, {3, 2}]];
	pars = Append[#,"*"]&/@pars;
	pars = Join[{msgEnum[[i,1]]}, pars];
	AppendTo[parList, pars];
];

(* net and layer par. lists are extra -> remove them before unfolding*)
(*parList = RemoveLayerParTypes[parList];
parList = RemoveNetParLayers[parList];*)
parList = RemoveParpattInMsg[parList, "LayerParameter", "*_param"];
parList = RemoveParpattInMsg[parList, "NetParameter", "layers"];
parList = RemoveParpattInMsg[parList, "SolverParameter", "*net_param"];

(* unfolding parameters with special type (message - group of pars.) *)
parListUnfold = {};
For[i = 1, i <= Length[msgs], i++, (*Length[msgs]*)
	AppendTo[parListUnfold, UnfoldMsgTypePars[parList[[i]], parList]];
];
(* currently there is no need to unfold more nested msg than 2 levels,
so this kinda nasty repeating is enough, since layers (LayerParameter) in
NetParameter is handled separately *)
parListUnfold2 = {};
For[i = 1, i <= Length[msgs], i++, (*Length[msgs]*)
	AppendTo[parListUnfold2, UnfoldMsgTypePars[parList[[i]], parListUnfold]];
];

paramLists = parListUnfold2
];
(* -------------------------------------------------------------------------- *)


RemoveParpattInMsg[pList_, msgName_, patt_] := Module[
{i, parList, msgPar, msgParPos, msgParPatt, p, depParP},
(* remove all layer type based parameters *)
parList = pList;
msgPar = Select[parList, StringMatchQ[#[[1]], msgName] &][[1]];
msgParPos = Flatten[Position[parList, msgPar]];
msgParPatt = Select[msgPar[[2;;]], StringMatchQ[#[[1]], patt] &];
(* if layer msg, save layer type based msgs *)
If[msgName == "LayerParameter",
  layerParTypeBased = msgParPatt;
];
p = {};
For[i=1, i <= Length[msgParPatt], i++,
  p = Join[p,Position[msgPar, msgParPatt[[i]]]];
];
(* also remove deprecated parameter V0LayerParameter, if present *)
depParP = Position[msgPar, "V0LayerParameter"];
If[Length[depParP] > 0,
  depParP = {{depParP[[1,1]]}}; (* and only 1. coord. is needed *)
  p = Join[p, depParP];
];
(* actual removal *)
msgPar = Delete[parList[[msgParPos]][[1]], p];

parList = ReplacePart[parList, msgParPos -> msgPar];
parList
];
(* -------------------------------------------------------------------------- *)


(* Unfold parameters referencing whole messages:
parList = {msgName, {p1Name, ""}, {p2Name, ""}},
msgPars = {p2Name, {p2TypeP1, ""}, p2TypeP2, ""}}
--\[Rule]
{msgName, {p1Name, "*b*"}, {p2Name, "*s*"}, {p2Name, {p2TypeP1, ""}, p2TypeP2, ""}}}.
There is actualy also type: {pName, pType, pValue} *)
UnfoldMsgTypePars[msgPars_, parList_] := Module[{pars, unfPars, unfp, j},
pars = msgPars;

unfPars = {};
For[j = 2, j <= Length[pars], j++,
	If[!IsBasicOrEnum[pars[[j, 2]]],
		pars[[j,3]] = strMsgType; (* set default value ... "*s*" *)
		unfp = Select[parList, StringMatchQ[#[[1]], pars[[j, 2]]] &][[1]];
		(* associate the new list with name of parameter rather then the type *)
		unfp[[1]] = pars[[j,1]];
		AppendTo[unfPars, unfp];
		,
		pars[[j,3]] = strBasicType; (* set default value ... "*b*" *)
  ];
];

AppendTo[pars, unfPars]
];
(* -------------------------------------------------------------------------- *)


(* Returns names and messages included in layer based on its type. *)
GetLayerParTypeBased[] := layerParTypeBased;
(* -------------------------------------------------------------------------- *)


cblasTest=LibraryFunctionLoad["libcaffeLink", "cblasTest", {}, "Void"];

initCaffeLinkLL=LibraryFunctionLoad["libcaffeLink", "initCaffeLink", {"Boolean", "Boolean", Integer}, "Void"];
prepareNetFile = LibraryFunctionLoad["libcaffeLink", "prepareNetFile", {"UTF8String"}, "Void"];
prepareNetString = LibraryFunctionLoad["libcaffeLink", "prepareNetString", {"UTF8String"}, "Void"];
loadNet = LibraryFunctionLoad["libcaffeLink", "loadNet", {"UTF8String"}, "Void"];
exportNet = LibraryFunctionLoad["libcaffeLink", "exportNet", {"UTF8String"}, "Void"];

testNet=LibraryFunctionLoad["libcaffeLink", "testNet", {}, "Void"];
trainNetString=LibraryFunctionLoad["libcaffeLink", "trainNetString",{"UTF8String"},"Void"];
trainNetFile=LibraryFunctionLoad["libcaffeLink", "trainNetFile",{"UTF8String"},"Void"];
trainNetSnapshotString=LibraryFunctionLoad["libcaffeLink", "trainNetSnapshotString",{"UTF8String","UTF8String"},"Void"];
trainNetSnapshotFile=LibraryFunctionLoad["libcaffeLink", "trainNetSnapshotFile",{"UTF8String","UTF8String"},"Void"];
trainNetWeightsString=LibraryFunctionLoad["libcaffeLink", "trainNetWeightsString",{"UTF8String","UTF8String"},"Void"];
trainNetWeightsFile=LibraryFunctionLoad["libcaffeLink", "trainNetWeightsFile",{"UTF8String","UTF8String"},"Void"];

printNetInfo = LibraryFunctionLoad["libcaffeLink","printNetInfo",{},"Void"];
printWorkingPath = LibraryFunctionLoad["libcaffeLink","printWorkingPath",{},"Void"];

getLayerNum = LibraryFunctionLoad["libcaffeLink","getLayerNum",{},Integer];
getTopBlobSize = LibraryFunctionLoad["libcaffeLink","getTopBlobSize",{Integer},{Integer,1}];
getBottomBlobSize = LibraryFunctionLoad["libcaffeLink","getBottomBlobSize",{Integer},{Integer,1}];
getParamBlobSize = LibraryFunctionLoad["libcaffeLink","getParamBlobSize",{Integer},{Integer,1}];

getTopBlob= LibraryFunctionLoad["libcaffeLink","getTopBlob",{Integer,Integer},{Real,1}];
getBottomBlob= LibraryFunctionLoad["libcaffeLink","getBottomBlob",{Integer,Integer},{Real,1}];
getParamBlob= LibraryFunctionLoad["libcaffeLink","getParamBlob",{Integer,Integer},{Real,1}];

setTopBlob= LibraryFunctionLoad["libcaffeLink","setTopBlob",{{Real,1},Integer,Integer},"Void"];
setBottomBlob= LibraryFunctionLoad["libcaffeLink","setTopBlob",{{Real,1},Integer,Integer},"Void"];
setParamBlob= LibraryFunctionLoad["libcaffeLink","setParamBlob",{{Real,1},Integer,Integer},"Void"];
setInput= LibraryFunctionLoad["libcaffeLink","setInput",{{Real,1}},"Void"];
(* -------------------------------------------------------------------------- *)


initCaffeLink[a_, b_, c_] := Module[{useDouble, gpu, devid},
useDouble = Replace["use double", {a, b, c}];
gpu = Replace["GPU mode", {a, b, c}];
devid = Replace["GPU device", {a, b, c}];
Print[{useDouble, gpu, devid}];
initCaffeLinkLL[useDouble, gpu, devid]
];
(* -------------------------------------------------------------------------- *)



End[]
EndPackage[]
