(* ::Package:: *)

BeginPackage["CaffeLink`"]

initCaffeLinkModule::usage = "initCaffeLinkModule[path] initializes module from caffe.proto file."
newNet::usage = "newNetOld[netParameters] returns net internal reprezentation, 'netParameters' format is {NetParam01, NetParam02, {{Layer01Param01, Layer01Param02, ..}, {Layer02Param01, Layer02Param02, ..}}}."
newSolver::usage = "newSolver[solverParameters] return solver internal reprezentation, 'solverParameters' format is {SolverParam01, SolverParam02, ...}."
solverAddNetParam::usage = "solverAddNetParam[solverParametersString, netParametersString] adds net definition to solver as net_param field. Returns resulting solver parameter string."
getParamString::usage = "getParamString[net] generates protobuffer string from net compatible with Caffe."

cblasTest::usage = "Test cblas stability."

initCaffeLink::usage = "initCaffeLink[UseDouble, UseGPU, GPUDevice] initializes CaffeLink library with data type double or float, sets GPU or CPU mode and device ID for GPU mode, default is 0."
prepareNetFile::usage = "prepareNetFile[path] sets up a new net from protobuffer file on given path."
prepareNetString::usage = "prepareNetString[string] sets up a new net from given protobuffer string."
loadNet::usage = "loadNet[path] loads previously exported or snapshoted net after the same net was prepared using prepareNet*."
exportNet::usage = "exportNet[path] exports previously loaded net to selected path."

evaluateNet::usage = "evaluateNet[] runs loaded net in test phase."
trainNetString::usage = "trainNetString[solverParam] trains new net with solver protobuffer parameters given in string."
trainNetFile::usage = "trainNetFile[path] trains new net with solver protobuffer parameters taken from file."
trainNetSnapshotString::usage = "trainNetSnapshotString[solverParam, pathState] continues training from solver state file on 'pathState'. Solver parameters given as string."
trainNetSnapshotFile::usage = "trainNetSnapshotFile[path, pathState] continues training from solver state file on 'pathState'. Solver parameters given as 'path' to file."
trainNetWeightsString::usage = "trainNetWeightsString[solverParam, pathWeights] finetunes net on path 'pathWeights'. Solver parameters given as string."
trainNetWeightsFile::usage = "trainNetWeightsFile[path, pathWeights] finetunes net on path 'pathWeights'. Solver parameters given as 'path' to file."

printNetInfo::usage = "printNetInfo[] prints (in console, stdout) info about prepared net: structure, layers, sizes and counts of blobs, etc."
printWorkingPath::usage = "printWorkingPath[] prints (in console, stdout) root directory, executes 'pwd' command."

getLayerNum::usage = "getLayerNum[] returns number of layers in prepared net."
getTopBlobSize::usage = "getTopBlobSize[layer] returns array of top blob dimensions: {num, channels, height, width}, 4 elements for each blob, 'layer' can be layer index or name."
getBottomBlobSize::usage = "getBottomBlobSize[layer] returns array of bottom blob dimensions: {num, channels, height, width}, 4 elements for each blob, 'layer' can be layer index or name.."
getParamBlobSize::usage = "getParamBlobSize[layer] returns array of parameter blob dimensions: {num, channels, height, width}, 4 elements for each blob, 'layer' can be layer index or name.."

getTopBlob::usage = "getTopBlob[layer, blobIdx] returns data stored in top 'blobIdx'-th blob of layer with index or name 'layer'. Default 'blobIdx' is 0."
getBottomBlob::usage = "getBottomBlob[layer, blobIdx] returns data stored in bottom 'blobIdx'-th blob of layer with index or name 'layer'. Default 'blobIdx' is 0."
getParamBlob::usage = "getParamBlob[layer, blobIdx] returns data stored in parameter 'blobIdx'-th blob of layer with index or name 'layer'. Default 'blobIdx' is 0."

setTopBlob::usage = "setTopBlob[data, layer, blobIdx] inserts 'data' to top 'blobIdx'-th blob of layer with index or name 'layer'. Default 'blobIdx' is 0."
setBottomBlob::usage = "setBottomBlob[data, layer, blobIdx] inserts 'data' to bottom 'blobIdx'-th blob of layer with index or name 'layer'. Default 'blobIdx' is 0."
setParamBlob::usage = "setParamBlob[data, layer, blobIdx] inserts 'data' to parameter 'blobIdx'-th blob of layer with index or name 'layer'. Default 'blobIdx' is 0."
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
If[!FileExistsQ[caffeProto],
	Throw[StringJoin["file not found: ", caffeProto]];
];

clProto = CleanProto[caffeProto];
msgWithEnums = ParseEnumsInMsg[clProto];
globalEnums = ParseGlobalEnums[clProto];
ParseParamList[globalEnums, msgWithEnums, clProto];
Return[True];
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


solverAddNetParam[solver_, net_] := Module[{netp},

StringJoin["net_param: {\n", net, "}\n", solver]
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
prepareNetFileLL = LibraryFunctionLoad["libcaffeLink", "prepareNetFile", {"UTF8String"}, "Void"];
prepareNetString = LibraryFunctionLoad["libcaffeLink", "prepareNetString", {"UTF8String"}, "Void"];
loadNetLL = LibraryFunctionLoad["libcaffeLink", "loadNet", {"UTF8String"}, "Void"];
exportNet = LibraryFunctionLoad["libcaffeLink", "exportNet", {"UTF8String"}, "Void"];

evaluateNet=LibraryFunctionLoad["libcaffeLink", "testNet", {}, "Void"];
trainNetString=LibraryFunctionLoad["libcaffeLink", "trainNetString",{"UTF8String"},"Void"];
trainNetFileLL=LibraryFunctionLoad["libcaffeLink", "trainNetFile",{"UTF8String"},"Void"];
trainNetSnapshotStringLL=LibraryFunctionLoad["libcaffeLink", "trainNetSnapshotString",{"UTF8String","UTF8String"},"Void"];
trainNetSnapshotFileLL=LibraryFunctionLoad["libcaffeLink", "trainNetSnapshotFile",{"UTF8String","UTF8String"},"Void"];
trainNetWeightsStringLL=LibraryFunctionLoad["libcaffeLink", "trainNetWeightsString",{"UTF8String","UTF8String"},"Void"];
trainNetWeightsFileLL=LibraryFunctionLoad["libcaffeLink", "trainNetWeightsFile",{"UTF8String","UTF8String"},"Void"];

printNetInfo = LibraryFunctionLoad["libcaffeLink","printNetInfo",{},"Void"];
printWorkingPath = LibraryFunctionLoad["libcaffeLink","printWorkingPath",{},"Void"];

getLayerNum = LibraryFunctionLoad["libcaffeLink","getLayerNum",{},Integer];
getTopBlobSizeLL = LibraryFunctionLoad["libcaffeLink","getTopBlobSize",{Integer},{Integer,1}];
getBottomBlobSizeLL = LibraryFunctionLoad["libcaffeLink","getBottomBlobSize",{Integer},{Integer,1}];
getParamBlobSizeLL = LibraryFunctionLoad["libcaffeLink","getParamBlobSize",{Integer},{Integer,1}];
getTopBlobSizeLNameLL = LibraryFunctionLoad["libcaffeLink","getTopBlobSizeLName",{"UTF8String"},{Integer,1}];
getBottomBlobSizeLNameLL = LibraryFunctionLoad["libcaffeLink","getBottomBlobSizeLName",{"UTF8String"},{Integer,1}];
getParamBlobSizeLNameLL = LibraryFunctionLoad["libcaffeLink","getParamBlobSizeLName",{"UTF8String"},{Integer,1}];

getTopBlobLL = LibraryFunctionLoad["libcaffeLink","getTopBlob",{Integer,Integer},{Real,1}];
getBottomBlobLL = LibraryFunctionLoad["libcaffeLink","getBottomBlob",{Integer,Integer},{Real,1}];
getParamBlobLL = LibraryFunctionLoad["libcaffeLink","getParamBlob",{Integer,Integer},{Real,1}];
getTopBlobLNameLL = LibraryFunctionLoad["libcaffeLink","getTopBlobLName",{"UTF8String",Integer},{Real,1}];
getBottomBlobLNameLL = LibraryFunctionLoad["libcaffeLink","getBottomBlobLName",{"UTF8String",Integer},{Real,1}];
getParamBlobLNameLL = LibraryFunctionLoad["libcaffeLink","getParamBlobLName",{"UTF8String",Integer},{Real,1}];

setTopBlobLL = LibraryFunctionLoad["libcaffeLink","setTopBlob",{{Real,1,"Manual"},Integer,Integer},"Void"];
setBottomBlobLL = LibraryFunctionLoad["libcaffeLink","setBottomBlob",{{Real,1,"Manual"},Integer,Integer},"Void"];
setParamBlobLL = LibraryFunctionLoad["libcaffeLink","setParamBlob",{{Real,1,"Manual"},Integer,Integer},"Void"];

setTopBlobLNameLL = LibraryFunctionLoad["libcaffeLink","setTopBlobLName",{{Real,1,"Manual"},"UTF8String",Integer},"Void"];
setBottomBlobLNameLL = LibraryFunctionLoad["libcaffeLink","setBottomBlobLName",{{Real,1,"Manual"},"UTF8String",Integer},"Void"];
setParamBlobLNameLL = LibraryFunctionLoad["libcaffeLink","setParamBlobLName",{{Real,1,"Manual"},"UTF8String",Integer},"Void"];
setInput = LibraryFunctionLoad["libcaffeLink","setInput",{{Real,1,"Manual"}},"Void"];
(* -------------------------------------------------------------------------- *)


initCaffeLink[ra_, rb_, rc_:{"GPUDevice"->0}] := Module[{useDouble, useGpu, devid, res},
args = Flatten[{ra, rb, rc}];
useDouble = Replace["UseDouble", args];
useGpu = Replace["UseGPU", args];
devid = Replace["GPUDevice", args];

res = initCaffeLinkLL[useDouble, useGpu, devid];
If[res == Null, Return[True],
	Print[res];
	Return[False];
];
];
(* -------------------------------------------------------------------------- *)


prepareNetFile[path_] := Module[{},
If[!FileExistsQ[path],
	Throw[StringJoin["file not found: ", path]];
];
prepareNetFileLL[path]
];
(* -------------------------------------------------------------------------- *)


loadNet[path_] := Module[{},
If[!FileExistsQ[path],
	Throw[StringJoin["file not found: ", path]];
];
loadNetLL[path]
];
(* -------------------------------------------------------------------------- *)


trainNetFile[path_] := Module[{},
If[!FileExistsQ[path],
	Throw[StringJoin["file not found: ", path]];
];
trainNetFileLL[path]
];
(* -------------------------------------------------------------------------- *)


trainNetSnapshotString

trainNetSnapshotString[solver_, path_] := Module[{},
If[!FileExistsQ[path],
	Throw[StringJoin["file not found: ", path]];
];
trainNetSnapshotStringLL[solver, path]
];
(* -------------------------------------------------------------------------- *)


trainNetSnapshotFile[solverPath_, path_] := Module[{},
If[!FileExistsQ[path],
	Throw[StringJoin["file not found: ", path]];
];
If[!FileExistsQ[solverPath],
	Throw[StringJoin["file not found: ", solverPath]];
];
trainNetSnapshotFileLL[solverPath, path]
];
(* -------------------------------------------------------------------------- *)


trainNetWeightsString[solver_, path_] := Module[{},
If[!FileExistsQ[path],
	Throw[StringJoin["file not found: ", path]];
];
trainNetWeightsStringLL[solver, path]
];
(* -------------------------------------------------------------------------- *)


trainNetWeightsFile[solverPath_, path_] := Module[{},
If[!FileExistsQ[path],
	Throw[StringJoin["file not found: ", path]];
];
If[!FileExistsQ[solverPath],
	Throw[StringJoin["file not found: ", solverPath]];
];
trainNetWeightsFileLL[solverPath, path]
];
(* -------------------------------------------------------------------------- *)


getTopBlob[layer_, blob_:0] := Module[{},
If[StringQ[layer],
  Return[getTopBlobLNameLL[layer,blob]];
  ,
  Return[getTopBlobLL[layer,blob]];
];
];
(* -------------------------------------------------------------------------- *)


getParamBlob[layer_, blob_:0] := Module[{},
If[StringQ[layer],
  Return[getParamBlobLNameLL[layer,blob]];
  ,
  Return[getParamBlobLL[layer,blob]];
];
];
(* -------------------------------------------------------------------------- *)


getBottomBlob[layer_, blob_:0] := Module[{},
If[StringQ[layer],
  Return[getBottomBlobLNameLL[layer,blob]];
  ,
  Return[getBottomBlobLL[layer,blob]];
];
];
(* -------------------------------------------------------------------------- *)


getTopBlobSize[layer_] := Module[{},
If[StringQ[layer],
  Return[getTopBlobSizeLNameLL[layer]];
  ,
  Return[getTopBlobSizeLL[layer]];
];
];
(* -------------------------------------------------------------------------- *)


getBottomBlobSize[layer_] := Module[{},
If[StringQ[layer],
  Return[getBottomBlobSizeLNameLL[layer]];
  ,
  Return[getBottomBlobSizeLL[layer]];
];
];
(* -------------------------------------------------------------------------- *)


getParamBlobSize[layer_] := Module[{},
If[StringQ[layer],
  Return[getParamBlobSizeLNameLL[layer]];
  ,
  Return[getParamBlobSizeLL[layer]];
];
];
(* -------------------------------------------------------------------------- *)


setTopBlob[data_, layer_, blob_:0] := Module[{},
If[StringQ[layer],
  Return[setTopBlobLNameLL[data, layer, blob]];
  ,
  Return[setTopBlobLL[data, layer, blob]];
];
];
(* -------------------------------------------------------------------------- *)


setBottomBlob[data_, layer_, blob_:0] := Module[{},
If[StringQ[layer],
  Return[setBottomBlobLNameLL[data, layer, blob]];
  ,
  Return[setBottomBlobLL[data, layer, blob]];
];
];
(* -------------------------------------------------------------------------- *)


setParamBlob[data_, layer_, blob_:0] := Module[{},
If[StringQ[layer],
  Return[setParamBlobLNameLL[data, layer, blob]];
  ,
  Return[setParamBlobLL[data, layer, blob]];
];
];
(* -------------------------------------------------------------------------- *)


End[]
EndPackage[]
