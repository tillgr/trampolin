??
??
B
AssignVariableOp
resource
value"dtype"
dtypetype?
~
BiasAdd

value"T	
bias"T
output"T" 
Ttype:
2	"-
data_formatstringNHWC:
NHWCNCHW
8
Const
output"dtype"
valuetensor"
dtypetype
.
Identity

input"T
output"T"	
Ttype
q
MatMul
a"T
b"T
product"T"
transpose_abool( "
transpose_bbool( "
Ttype:

2	
e
MergeV2Checkpoints
checkpoint_prefixes
destination_prefix"
delete_old_dirsbool(?

NoOp
M
Pack
values"T*N
output"T"
Nint(0"	
Ttype"
axisint 
C
Placeholder
output"dtype"
dtypetype"
shapeshape:
@
ReadVariableOp
resource
value"dtype"
dtypetype?
E
Relu
features"T
activations"T"
Ttype:
2	
o
	RestoreV2

prefix
tensor_names
shape_and_slices
tensors2dtypes"
dtypes
list(type)(0?
l
SaveV2

prefix
tensor_names
shape_and_slices
tensors2dtypes"
dtypes
list(type)(0?
?
Select
	condition

t"T
e"T
output"T"	
Ttype
H
ShardedFilename
basename	
shard

num_shards
filename
9
Softmax
logits"T
softmax"T"
Ttype:
2
?
StatefulPartitionedCall
args2Tin
output2Tout"
Tin
list(type)("
Tout
list(type)("	
ffunc"
configstring "
config_protostring "
executor_typestring ?
@
StaticRegexFullMatch	
input

output
"
patternstring
N

StringJoin
inputs*N

output"
Nint(0"
	separatorstring 
?
VarHandleOp
resource"
	containerstring "
shared_namestring "
dtypetype"
shapeshape"#
allowed_deviceslist(string)
 ?"serve*2.4.02v2.4.0-rc4-71-g582c8d236cb8??

}
dense_105/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:	F?*!
shared_namedense_105/kernel
v
$dense_105/kernel/Read/ReadVariableOpReadVariableOpdense_105/kernel*
_output_shapes
:	F?*
dtype0
u
dense_105/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*
shared_namedense_105/bias
n
"dense_105/bias/Read/ReadVariableOpReadVariableOpdense_105/bias*
_output_shapes	
:?*
dtype0
~
dense_106/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:
??*!
shared_namedense_106/kernel
w
$dense_106/kernel/Read/ReadVariableOpReadVariableOpdense_106/kernel* 
_output_shapes
:
??*
dtype0
u
dense_106/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*
shared_namedense_106/bias
n
"dense_106/bias/Read/ReadVariableOpReadVariableOpdense_106/bias*
_output_shapes	
:?*
dtype0
~
dense_107/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:
??*!
shared_namedense_107/kernel
w
$dense_107/kernel/Read/ReadVariableOpReadVariableOpdense_107/kernel* 
_output_shapes
:
??*
dtype0
u
dense_107/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*
shared_namedense_107/bias
n
"dense_107/bias/Read/ReadVariableOpReadVariableOpdense_107/bias*
_output_shapes	
:?*
dtype0
~
dense_108/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:
??*!
shared_namedense_108/kernel
w
$dense_108/kernel/Read/ReadVariableOpReadVariableOpdense_108/kernel* 
_output_shapes
:
??*
dtype0
u
dense_108/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*
shared_namedense_108/bias
n
"dense_108/bias/Read/ReadVariableOpReadVariableOpdense_108/bias*
_output_shapes	
:?*
dtype0
~
dense_109/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:
??*!
shared_namedense_109/kernel
w
$dense_109/kernel/Read/ReadVariableOpReadVariableOpdense_109/kernel* 
_output_shapes
:
??*
dtype0
u
dense_109/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*
shared_namedense_109/bias
n
"dense_109/bias/Read/ReadVariableOpReadVariableOpdense_109/bias*
_output_shapes	
:?*
dtype0
~
dense_110/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:
??*!
shared_namedense_110/kernel
w
$dense_110/kernel/Read/ReadVariableOpReadVariableOpdense_110/kernel* 
_output_shapes
:
??*
dtype0
u
dense_110/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*
shared_namedense_110/bias
n
"dense_110/bias/Read/ReadVariableOpReadVariableOpdense_110/bias*
_output_shapes	
:?*
dtype0
~
dense_111/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:
??*!
shared_namedense_111/kernel
w
$dense_111/kernel/Read/ReadVariableOpReadVariableOpdense_111/kernel* 
_output_shapes
:
??*
dtype0
u
dense_111/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*
shared_namedense_111/bias
n
"dense_111/bias/Read/ReadVariableOpReadVariableOpdense_111/bias*
_output_shapes	
:?*
dtype0
w
output/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:	?+*
shared_nameoutput/kernel
p
!output/kernel/Read/ReadVariableOpReadVariableOpoutput/kernel*
_output_shapes
:	?+*
dtype0
n
output/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:+*
shared_nameoutput/bias
g
output/bias/Read/ReadVariableOpReadVariableOpoutput/bias*
_output_shapes
:+*
dtype0
h

Nadam/iterVarHandleOp*
_output_shapes
: *
dtype0	*
shape: *
shared_name
Nadam/iter
a
Nadam/iter/Read/ReadVariableOpReadVariableOp
Nadam/iter*
_output_shapes
: *
dtype0	
l
Nadam/beta_1VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_nameNadam/beta_1
e
 Nadam/beta_1/Read/ReadVariableOpReadVariableOpNadam/beta_1*
_output_shapes
: *
dtype0
l
Nadam/beta_2VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_nameNadam/beta_2
e
 Nadam/beta_2/Read/ReadVariableOpReadVariableOpNadam/beta_2*
_output_shapes
: *
dtype0
j
Nadam/decayVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_nameNadam/decay
c
Nadam/decay/Read/ReadVariableOpReadVariableOpNadam/decay*
_output_shapes
: *
dtype0
z
Nadam/learning_rateVarHandleOp*
_output_shapes
: *
dtype0*
shape: *$
shared_nameNadam/learning_rate
s
'Nadam/learning_rate/Read/ReadVariableOpReadVariableOpNadam/learning_rate*
_output_shapes
: *
dtype0
|
Nadam/momentum_cacheVarHandleOp*
_output_shapes
: *
dtype0*
shape: *%
shared_nameNadam/momentum_cache
u
(Nadam/momentum_cache/Read/ReadVariableOpReadVariableOpNadam/momentum_cache*
_output_shapes
: *
dtype0
^
totalVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_nametotal
W
total/Read/ReadVariableOpReadVariableOptotal*
_output_shapes
: *
dtype0
^
countVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_namecount
W
count/Read/ReadVariableOpReadVariableOpcount*
_output_shapes
: *
dtype0
b
total_1VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_name	total_1
[
total_1/Read/ReadVariableOpReadVariableOptotal_1*
_output_shapes
: *
dtype0
b
count_1VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_name	count_1
[
count_1/Read/ReadVariableOpReadVariableOpcount_1*
_output_shapes
: *
dtype0
n
accumulatorVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_nameaccumulator
g
accumulator/Read/ReadVariableOpReadVariableOpaccumulator*
_output_shapes
:*
dtype0
r
accumulator_1VarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_nameaccumulator_1
k
!accumulator_1/Read/ReadVariableOpReadVariableOpaccumulator_1*
_output_shapes
:*
dtype0
r
accumulator_2VarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_nameaccumulator_2
k
!accumulator_2/Read/ReadVariableOpReadVariableOpaccumulator_2*
_output_shapes
:*
dtype0
r
accumulator_3VarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_nameaccumulator_3
k
!accumulator_3/Read/ReadVariableOpReadVariableOpaccumulator_3*
_output_shapes
:*
dtype0
?
Nadam/dense_105/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:	F?*)
shared_nameNadam/dense_105/kernel/m
?
,Nadam/dense_105/kernel/m/Read/ReadVariableOpReadVariableOpNadam/dense_105/kernel/m*
_output_shapes
:	F?*
dtype0
?
Nadam/dense_105/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*'
shared_nameNadam/dense_105/bias/m
~
*Nadam/dense_105/bias/m/Read/ReadVariableOpReadVariableOpNadam/dense_105/bias/m*
_output_shapes	
:?*
dtype0
?
Nadam/dense_106/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:
??*)
shared_nameNadam/dense_106/kernel/m
?
,Nadam/dense_106/kernel/m/Read/ReadVariableOpReadVariableOpNadam/dense_106/kernel/m* 
_output_shapes
:
??*
dtype0
?
Nadam/dense_106/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*'
shared_nameNadam/dense_106/bias/m
~
*Nadam/dense_106/bias/m/Read/ReadVariableOpReadVariableOpNadam/dense_106/bias/m*
_output_shapes	
:?*
dtype0
?
Nadam/dense_107/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:
??*)
shared_nameNadam/dense_107/kernel/m
?
,Nadam/dense_107/kernel/m/Read/ReadVariableOpReadVariableOpNadam/dense_107/kernel/m* 
_output_shapes
:
??*
dtype0
?
Nadam/dense_107/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*'
shared_nameNadam/dense_107/bias/m
~
*Nadam/dense_107/bias/m/Read/ReadVariableOpReadVariableOpNadam/dense_107/bias/m*
_output_shapes	
:?*
dtype0
?
Nadam/dense_108/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:
??*)
shared_nameNadam/dense_108/kernel/m
?
,Nadam/dense_108/kernel/m/Read/ReadVariableOpReadVariableOpNadam/dense_108/kernel/m* 
_output_shapes
:
??*
dtype0
?
Nadam/dense_108/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*'
shared_nameNadam/dense_108/bias/m
~
*Nadam/dense_108/bias/m/Read/ReadVariableOpReadVariableOpNadam/dense_108/bias/m*
_output_shapes	
:?*
dtype0
?
Nadam/dense_109/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:
??*)
shared_nameNadam/dense_109/kernel/m
?
,Nadam/dense_109/kernel/m/Read/ReadVariableOpReadVariableOpNadam/dense_109/kernel/m* 
_output_shapes
:
??*
dtype0
?
Nadam/dense_109/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*'
shared_nameNadam/dense_109/bias/m
~
*Nadam/dense_109/bias/m/Read/ReadVariableOpReadVariableOpNadam/dense_109/bias/m*
_output_shapes	
:?*
dtype0
?
Nadam/dense_110/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:
??*)
shared_nameNadam/dense_110/kernel/m
?
,Nadam/dense_110/kernel/m/Read/ReadVariableOpReadVariableOpNadam/dense_110/kernel/m* 
_output_shapes
:
??*
dtype0
?
Nadam/dense_110/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*'
shared_nameNadam/dense_110/bias/m
~
*Nadam/dense_110/bias/m/Read/ReadVariableOpReadVariableOpNadam/dense_110/bias/m*
_output_shapes	
:?*
dtype0
?
Nadam/dense_111/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:
??*)
shared_nameNadam/dense_111/kernel/m
?
,Nadam/dense_111/kernel/m/Read/ReadVariableOpReadVariableOpNadam/dense_111/kernel/m* 
_output_shapes
:
??*
dtype0
?
Nadam/dense_111/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*'
shared_nameNadam/dense_111/bias/m
~
*Nadam/dense_111/bias/m/Read/ReadVariableOpReadVariableOpNadam/dense_111/bias/m*
_output_shapes	
:?*
dtype0
?
Nadam/output/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:	?+*&
shared_nameNadam/output/kernel/m
?
)Nadam/output/kernel/m/Read/ReadVariableOpReadVariableOpNadam/output/kernel/m*
_output_shapes
:	?+*
dtype0
~
Nadam/output/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:+*$
shared_nameNadam/output/bias/m
w
'Nadam/output/bias/m/Read/ReadVariableOpReadVariableOpNadam/output/bias/m*
_output_shapes
:+*
dtype0
?
Nadam/dense_105/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:	F?*)
shared_nameNadam/dense_105/kernel/v
?
,Nadam/dense_105/kernel/v/Read/ReadVariableOpReadVariableOpNadam/dense_105/kernel/v*
_output_shapes
:	F?*
dtype0
?
Nadam/dense_105/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*'
shared_nameNadam/dense_105/bias/v
~
*Nadam/dense_105/bias/v/Read/ReadVariableOpReadVariableOpNadam/dense_105/bias/v*
_output_shapes	
:?*
dtype0
?
Nadam/dense_106/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:
??*)
shared_nameNadam/dense_106/kernel/v
?
,Nadam/dense_106/kernel/v/Read/ReadVariableOpReadVariableOpNadam/dense_106/kernel/v* 
_output_shapes
:
??*
dtype0
?
Nadam/dense_106/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*'
shared_nameNadam/dense_106/bias/v
~
*Nadam/dense_106/bias/v/Read/ReadVariableOpReadVariableOpNadam/dense_106/bias/v*
_output_shapes	
:?*
dtype0
?
Nadam/dense_107/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:
??*)
shared_nameNadam/dense_107/kernel/v
?
,Nadam/dense_107/kernel/v/Read/ReadVariableOpReadVariableOpNadam/dense_107/kernel/v* 
_output_shapes
:
??*
dtype0
?
Nadam/dense_107/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*'
shared_nameNadam/dense_107/bias/v
~
*Nadam/dense_107/bias/v/Read/ReadVariableOpReadVariableOpNadam/dense_107/bias/v*
_output_shapes	
:?*
dtype0
?
Nadam/dense_108/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:
??*)
shared_nameNadam/dense_108/kernel/v
?
,Nadam/dense_108/kernel/v/Read/ReadVariableOpReadVariableOpNadam/dense_108/kernel/v* 
_output_shapes
:
??*
dtype0
?
Nadam/dense_108/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*'
shared_nameNadam/dense_108/bias/v
~
*Nadam/dense_108/bias/v/Read/ReadVariableOpReadVariableOpNadam/dense_108/bias/v*
_output_shapes	
:?*
dtype0
?
Nadam/dense_109/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:
??*)
shared_nameNadam/dense_109/kernel/v
?
,Nadam/dense_109/kernel/v/Read/ReadVariableOpReadVariableOpNadam/dense_109/kernel/v* 
_output_shapes
:
??*
dtype0
?
Nadam/dense_109/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*'
shared_nameNadam/dense_109/bias/v
~
*Nadam/dense_109/bias/v/Read/ReadVariableOpReadVariableOpNadam/dense_109/bias/v*
_output_shapes	
:?*
dtype0
?
Nadam/dense_110/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:
??*)
shared_nameNadam/dense_110/kernel/v
?
,Nadam/dense_110/kernel/v/Read/ReadVariableOpReadVariableOpNadam/dense_110/kernel/v* 
_output_shapes
:
??*
dtype0
?
Nadam/dense_110/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*'
shared_nameNadam/dense_110/bias/v
~
*Nadam/dense_110/bias/v/Read/ReadVariableOpReadVariableOpNadam/dense_110/bias/v*
_output_shapes	
:?*
dtype0
?
Nadam/dense_111/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:
??*)
shared_nameNadam/dense_111/kernel/v
?
,Nadam/dense_111/kernel/v/Read/ReadVariableOpReadVariableOpNadam/dense_111/kernel/v* 
_output_shapes
:
??*
dtype0
?
Nadam/dense_111/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*'
shared_nameNadam/dense_111/bias/v
~
*Nadam/dense_111/bias/v/Read/ReadVariableOpReadVariableOpNadam/dense_111/bias/v*
_output_shapes	
:?*
dtype0
?
Nadam/output/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:	?+*&
shared_nameNadam/output/kernel/v
?
)Nadam/output/kernel/v/Read/ReadVariableOpReadVariableOpNadam/output/kernel/v*
_output_shapes
:	?+*
dtype0
~
Nadam/output/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:+*$
shared_nameNadam/output/bias/v
w
'Nadam/output/bias/v/Read/ReadVariableOpReadVariableOpNadam/output/bias/v*
_output_shapes
:+*
dtype0

NoOpNoOp
?Y
ConstConst"/device:CPU:0*
_output_shapes
: *
dtype0*?X
value?XB?X B?X
?
layer-0
layer_with_weights-0
layer-1
layer_with_weights-1
layer-2
layer_with_weights-2
layer-3
layer_with_weights-3
layer-4
layer_with_weights-4
layer-5
layer_with_weights-5
layer-6
layer_with_weights-6
layer-7
	layer_with_weights-7
	layer-8

	optimizer
regularization_losses
	variables
trainable_variables
	keras_api

signatures
 
h

kernel
bias
regularization_losses
	variables
trainable_variables
	keras_api
h

kernel
bias
regularization_losses
	variables
trainable_variables
	keras_api
h

kernel
bias
regularization_losses
	variables
 trainable_variables
!	keras_api
h

"kernel
#bias
$regularization_losses
%	variables
&trainable_variables
'	keras_api
h

(kernel
)bias
*regularization_losses
+	variables
,trainable_variables
-	keras_api
h

.kernel
/bias
0regularization_losses
1	variables
2trainable_variables
3	keras_api
h

4kernel
5bias
6regularization_losses
7	variables
8trainable_variables
9	keras_api
h

:kernel
;bias
<regularization_losses
=	variables
>trainable_variables
?	keras_api
?
@iter

Abeta_1

Bbeta_2
	Cdecay
Dlearning_rate
Emomentum_cachem?m?m?m?m?m?"m?#m?(m?)m?.m?/m?4m?5m?:m?;m?v?v?v?v?v?v?"v?#v?(v?)v?.v?/v?4v?5v?:v?;v?
 
v
0
1
2
3
4
5
"6
#7
(8
)9
.10
/11
412
513
:14
;15
v
0
1
2
3
4
5
"6
#7
(8
)9
.10
/11
412
513
:14
;15
?

Flayers
Gmetrics
Hnon_trainable_variables
regularization_losses
Ilayer_regularization_losses
Jlayer_metrics
	variables
trainable_variables
 
\Z
VARIABLE_VALUEdense_105/kernel6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUE
XV
VARIABLE_VALUEdense_105/bias4layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUE
 

0
1

0
1
?

Klayers
Lnon_trainable_variables
regularization_losses
	variables
Mlayer_regularization_losses
Nlayer_metrics
Ometrics
trainable_variables
\Z
VARIABLE_VALUEdense_106/kernel6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUE
XV
VARIABLE_VALUEdense_106/bias4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUE
 

0
1

0
1
?

Players
Qnon_trainable_variables
regularization_losses
	variables
Rlayer_regularization_losses
Slayer_metrics
Tmetrics
trainable_variables
\Z
VARIABLE_VALUEdense_107/kernel6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUE
XV
VARIABLE_VALUEdense_107/bias4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUE
 

0
1

0
1
?

Ulayers
Vnon_trainable_variables
regularization_losses
	variables
Wlayer_regularization_losses
Xlayer_metrics
Ymetrics
 trainable_variables
\Z
VARIABLE_VALUEdense_108/kernel6layer_with_weights-3/kernel/.ATTRIBUTES/VARIABLE_VALUE
XV
VARIABLE_VALUEdense_108/bias4layer_with_weights-3/bias/.ATTRIBUTES/VARIABLE_VALUE
 

"0
#1

"0
#1
?

Zlayers
[non_trainable_variables
$regularization_losses
%	variables
\layer_regularization_losses
]layer_metrics
^metrics
&trainable_variables
\Z
VARIABLE_VALUEdense_109/kernel6layer_with_weights-4/kernel/.ATTRIBUTES/VARIABLE_VALUE
XV
VARIABLE_VALUEdense_109/bias4layer_with_weights-4/bias/.ATTRIBUTES/VARIABLE_VALUE
 

(0
)1

(0
)1
?

_layers
`non_trainable_variables
*regularization_losses
+	variables
alayer_regularization_losses
blayer_metrics
cmetrics
,trainable_variables
\Z
VARIABLE_VALUEdense_110/kernel6layer_with_weights-5/kernel/.ATTRIBUTES/VARIABLE_VALUE
XV
VARIABLE_VALUEdense_110/bias4layer_with_weights-5/bias/.ATTRIBUTES/VARIABLE_VALUE
 

.0
/1

.0
/1
?

dlayers
enon_trainable_variables
0regularization_losses
1	variables
flayer_regularization_losses
glayer_metrics
hmetrics
2trainable_variables
\Z
VARIABLE_VALUEdense_111/kernel6layer_with_weights-6/kernel/.ATTRIBUTES/VARIABLE_VALUE
XV
VARIABLE_VALUEdense_111/bias4layer_with_weights-6/bias/.ATTRIBUTES/VARIABLE_VALUE
 

40
51

40
51
?

ilayers
jnon_trainable_variables
6regularization_losses
7	variables
klayer_regularization_losses
llayer_metrics
mmetrics
8trainable_variables
YW
VARIABLE_VALUEoutput/kernel6layer_with_weights-7/kernel/.ATTRIBUTES/VARIABLE_VALUE
US
VARIABLE_VALUEoutput/bias4layer_with_weights-7/bias/.ATTRIBUTES/VARIABLE_VALUE
 

:0
;1

:0
;1
?

nlayers
onon_trainable_variables
<regularization_losses
=	variables
player_regularization_losses
qlayer_metrics
rmetrics
>trainable_variables
IG
VARIABLE_VALUE
Nadam/iter)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUE
MK
VARIABLE_VALUENadam/beta_1+optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUE
MK
VARIABLE_VALUENadam/beta_2+optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUE
KI
VARIABLE_VALUENadam/decay*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUE
[Y
VARIABLE_VALUENadam/learning_rate2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUE
][
VARIABLE_VALUENadam/momentum_cache3optimizer/momentum_cache/.ATTRIBUTES/VARIABLE_VALUE
?
0
1
2
3
4
5
6
7
	8
*
s0
t1
u2
v3
w4
x5
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
4
	ytotal
	zcount
{	variables
|	keras_api
F
	}total
	~count

_fn_kwargs
?	variables
?	keras_api
C
?
thresholds
?accumulator
?	variables
?	keras_api
C
?
thresholds
?accumulator
?	variables
?	keras_api
C
?
thresholds
?accumulator
?	variables
?	keras_api
C
?
thresholds
?accumulator
?	variables
?	keras_api
OM
VARIABLE_VALUEtotal4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUE
OM
VARIABLE_VALUEcount4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUE

y0
z1

{	variables
QO
VARIABLE_VALUEtotal_14keras_api/metrics/1/total/.ATTRIBUTES/VARIABLE_VALUE
QO
VARIABLE_VALUEcount_14keras_api/metrics/1/count/.ATTRIBUTES/VARIABLE_VALUE
 

}0
~1

?	variables
 
[Y
VARIABLE_VALUEaccumulator:keras_api/metrics/2/accumulator/.ATTRIBUTES/VARIABLE_VALUE

?0

?	variables
 
][
VARIABLE_VALUEaccumulator_1:keras_api/metrics/3/accumulator/.ATTRIBUTES/VARIABLE_VALUE

?0

?	variables
 
][
VARIABLE_VALUEaccumulator_2:keras_api/metrics/4/accumulator/.ATTRIBUTES/VARIABLE_VALUE

?0

?	variables
 
][
VARIABLE_VALUEaccumulator_3:keras_api/metrics/5/accumulator/.ATTRIBUTES/VARIABLE_VALUE

?0

?	variables
?~
VARIABLE_VALUENadam/dense_105/kernel/mRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
|z
VARIABLE_VALUENadam/dense_105/bias/mPlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
?~
VARIABLE_VALUENadam/dense_106/kernel/mRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
|z
VARIABLE_VALUENadam/dense_106/bias/mPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
?~
VARIABLE_VALUENadam/dense_107/kernel/mRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
|z
VARIABLE_VALUENadam/dense_107/bias/mPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
?~
VARIABLE_VALUENadam/dense_108/kernel/mRlayer_with_weights-3/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
|z
VARIABLE_VALUENadam/dense_108/bias/mPlayer_with_weights-3/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
?~
VARIABLE_VALUENadam/dense_109/kernel/mRlayer_with_weights-4/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
|z
VARIABLE_VALUENadam/dense_109/bias/mPlayer_with_weights-4/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
?~
VARIABLE_VALUENadam/dense_110/kernel/mRlayer_with_weights-5/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
|z
VARIABLE_VALUENadam/dense_110/bias/mPlayer_with_weights-5/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
?~
VARIABLE_VALUENadam/dense_111/kernel/mRlayer_with_weights-6/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
|z
VARIABLE_VALUENadam/dense_111/bias/mPlayer_with_weights-6/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
}{
VARIABLE_VALUENadam/output/kernel/mRlayer_with_weights-7/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
yw
VARIABLE_VALUENadam/output/bias/mPlayer_with_weights-7/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
?~
VARIABLE_VALUENadam/dense_105/kernel/vRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
|z
VARIABLE_VALUENadam/dense_105/bias/vPlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
?~
VARIABLE_VALUENadam/dense_106/kernel/vRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
|z
VARIABLE_VALUENadam/dense_106/bias/vPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
?~
VARIABLE_VALUENadam/dense_107/kernel/vRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
|z
VARIABLE_VALUENadam/dense_107/bias/vPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
?~
VARIABLE_VALUENadam/dense_108/kernel/vRlayer_with_weights-3/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
|z
VARIABLE_VALUENadam/dense_108/bias/vPlayer_with_weights-3/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
?~
VARIABLE_VALUENadam/dense_109/kernel/vRlayer_with_weights-4/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
|z
VARIABLE_VALUENadam/dense_109/bias/vPlayer_with_weights-4/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
?~
VARIABLE_VALUENadam/dense_110/kernel/vRlayer_with_weights-5/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
|z
VARIABLE_VALUENadam/dense_110/bias/vPlayer_with_weights-5/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
?~
VARIABLE_VALUENadam/dense_111/kernel/vRlayer_with_weights-6/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
|z
VARIABLE_VALUENadam/dense_111/bias/vPlayer_with_weights-6/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
}{
VARIABLE_VALUENadam/output/kernel/vRlayer_with_weights-7/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
yw
VARIABLE_VALUENadam/output/bias/vPlayer_with_weights-7/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
~
serving_default_first_inputPlaceholder*'
_output_shapes
:?????????F*
dtype0*
shape:?????????F
?
StatefulPartitionedCallStatefulPartitionedCallserving_default_first_inputdense_105/kerneldense_105/biasdense_106/kerneldense_106/biasdense_107/kerneldense_107/biasdense_108/kerneldense_108/biasdense_109/kerneldense_109/biasdense_110/kerneldense_110/biasdense_111/kerneldense_111/biasoutput/kerneloutput/bias*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????+*2
_read_only_resource_inputs
	
*-
config_proto

CPU

GPU 2J 8? *-
f(R&
$__inference_signature_wrapper_675225
O
saver_filenamePlaceholder*
_output_shapes
: *
dtype0*
shape: 
?
StatefulPartitionedCall_1StatefulPartitionedCallsaver_filename$dense_105/kernel/Read/ReadVariableOp"dense_105/bias/Read/ReadVariableOp$dense_106/kernel/Read/ReadVariableOp"dense_106/bias/Read/ReadVariableOp$dense_107/kernel/Read/ReadVariableOp"dense_107/bias/Read/ReadVariableOp$dense_108/kernel/Read/ReadVariableOp"dense_108/bias/Read/ReadVariableOp$dense_109/kernel/Read/ReadVariableOp"dense_109/bias/Read/ReadVariableOp$dense_110/kernel/Read/ReadVariableOp"dense_110/bias/Read/ReadVariableOp$dense_111/kernel/Read/ReadVariableOp"dense_111/bias/Read/ReadVariableOp!output/kernel/Read/ReadVariableOpoutput/bias/Read/ReadVariableOpNadam/iter/Read/ReadVariableOp Nadam/beta_1/Read/ReadVariableOp Nadam/beta_2/Read/ReadVariableOpNadam/decay/Read/ReadVariableOp'Nadam/learning_rate/Read/ReadVariableOp(Nadam/momentum_cache/Read/ReadVariableOptotal/Read/ReadVariableOpcount/Read/ReadVariableOptotal_1/Read/ReadVariableOpcount_1/Read/ReadVariableOpaccumulator/Read/ReadVariableOp!accumulator_1/Read/ReadVariableOp!accumulator_2/Read/ReadVariableOp!accumulator_3/Read/ReadVariableOp,Nadam/dense_105/kernel/m/Read/ReadVariableOp*Nadam/dense_105/bias/m/Read/ReadVariableOp,Nadam/dense_106/kernel/m/Read/ReadVariableOp*Nadam/dense_106/bias/m/Read/ReadVariableOp,Nadam/dense_107/kernel/m/Read/ReadVariableOp*Nadam/dense_107/bias/m/Read/ReadVariableOp,Nadam/dense_108/kernel/m/Read/ReadVariableOp*Nadam/dense_108/bias/m/Read/ReadVariableOp,Nadam/dense_109/kernel/m/Read/ReadVariableOp*Nadam/dense_109/bias/m/Read/ReadVariableOp,Nadam/dense_110/kernel/m/Read/ReadVariableOp*Nadam/dense_110/bias/m/Read/ReadVariableOp,Nadam/dense_111/kernel/m/Read/ReadVariableOp*Nadam/dense_111/bias/m/Read/ReadVariableOp)Nadam/output/kernel/m/Read/ReadVariableOp'Nadam/output/bias/m/Read/ReadVariableOp,Nadam/dense_105/kernel/v/Read/ReadVariableOp*Nadam/dense_105/bias/v/Read/ReadVariableOp,Nadam/dense_106/kernel/v/Read/ReadVariableOp*Nadam/dense_106/bias/v/Read/ReadVariableOp,Nadam/dense_107/kernel/v/Read/ReadVariableOp*Nadam/dense_107/bias/v/Read/ReadVariableOp,Nadam/dense_108/kernel/v/Read/ReadVariableOp*Nadam/dense_108/bias/v/Read/ReadVariableOp,Nadam/dense_109/kernel/v/Read/ReadVariableOp*Nadam/dense_109/bias/v/Read/ReadVariableOp,Nadam/dense_110/kernel/v/Read/ReadVariableOp*Nadam/dense_110/bias/v/Read/ReadVariableOp,Nadam/dense_111/kernel/v/Read/ReadVariableOp*Nadam/dense_111/bias/v/Read/ReadVariableOp)Nadam/output/kernel/v/Read/ReadVariableOp'Nadam/output/bias/v/Read/ReadVariableOpConst*K
TinD
B2@	*
Tout
2*
_collective_manager_ids
 *
_output_shapes
: * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *(
f#R!
__inference__traced_save_675788
?
StatefulPartitionedCall_2StatefulPartitionedCallsaver_filenamedense_105/kerneldense_105/biasdense_106/kerneldense_106/biasdense_107/kerneldense_107/biasdense_108/kerneldense_108/biasdense_109/kerneldense_109/biasdense_110/kerneldense_110/biasdense_111/kerneldense_111/biasoutput/kerneloutput/bias
Nadam/iterNadam/beta_1Nadam/beta_2Nadam/decayNadam/learning_rateNadam/momentum_cachetotalcounttotal_1count_1accumulatoraccumulator_1accumulator_2accumulator_3Nadam/dense_105/kernel/mNadam/dense_105/bias/mNadam/dense_106/kernel/mNadam/dense_106/bias/mNadam/dense_107/kernel/mNadam/dense_107/bias/mNadam/dense_108/kernel/mNadam/dense_108/bias/mNadam/dense_109/kernel/mNadam/dense_109/bias/mNadam/dense_110/kernel/mNadam/dense_110/bias/mNadam/dense_111/kernel/mNadam/dense_111/bias/mNadam/output/kernel/mNadam/output/bias/mNadam/dense_105/kernel/vNadam/dense_105/bias/vNadam/dense_106/kernel/vNadam/dense_106/bias/vNadam/dense_107/kernel/vNadam/dense_107/bias/vNadam/dense_108/kernel/vNadam/dense_108/bias/vNadam/dense_109/kernel/vNadam/dense_109/bias/vNadam/dense_110/kernel/vNadam/dense_110/bias/vNadam/dense_111/kernel/vNadam/dense_111/bias/vNadam/output/kernel/vNadam/output/bias/v*J
TinC
A2?*
Tout
2*
_collective_manager_ids
 *
_output_shapes
: * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *+
f&R$
"__inference__traced_restore_675984??
?

?
)__inference_model_15_layer_call_fn_675382

inputs
unknown
	unknown_0
	unknown_1
	unknown_2
	unknown_3
	unknown_4
	unknown_5
	unknown_6
	unknown_7
	unknown_8
	unknown_9

unknown_10

unknown_11

unknown_12

unknown_13

unknown_14
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13
unknown_14*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????+*2
_read_only_resource_inputs
	
*-
config_proto

CPU

GPU 2J 8? *M
fHRF
D__inference_model_15_layer_call_and_return_conditional_losses_6750622
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:?????????+2

Identity"
identityIdentity:output:0*f
_input_shapesU
S:?????????F::::::::::::::::22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:?????????F
 
_user_specified_nameinputs
?

?
)__inference_model_15_layer_call_fn_675178
first_input
unknown
	unknown_0
	unknown_1
	unknown_2
	unknown_3
	unknown_4
	unknown_5
	unknown_6
	unknown_7
	unknown_8
	unknown_9

unknown_10

unknown_11

unknown_12

unknown_13

unknown_14
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallfirst_inputunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13
unknown_14*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????+*2
_read_only_resource_inputs
	
*-
config_proto

CPU

GPU 2J 8? *M
fHRF
D__inference_model_15_layer_call_and_return_conditional_losses_6751432
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:?????????+2

Identity"
identityIdentity:output:0*f
_input_shapesU
S:?????????F::::::::::::::::22
StatefulPartitionedCallStatefulPartitionedCall:T P
'
_output_shapes
:?????????F
%
_user_specified_namefirst_input
?

?
)__inference_model_15_layer_call_fn_675097
first_input
unknown
	unknown_0
	unknown_1
	unknown_2
	unknown_3
	unknown_4
	unknown_5
	unknown_6
	unknown_7
	unknown_8
	unknown_9

unknown_10

unknown_11

unknown_12

unknown_13

unknown_14
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallfirst_inputunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13
unknown_14*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????+*2
_read_only_resource_inputs
	
*-
config_proto

CPU

GPU 2J 8? *M
fHRF
D__inference_model_15_layer_call_and_return_conditional_losses_6750622
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:?????????+2

Identity"
identityIdentity:output:0*f
_input_shapesU
S:?????????F::::::::::::::::22
StatefulPartitionedCallStatefulPartitionedCall:T P
'
_output_shapes
:?????????F
%
_user_specified_namefirst_input
?	
?
E__inference_dense_107_layer_call_and_return_conditional_losses_674819

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identity??BiasAdd/ReadVariableOp?MatMul/ReadVariableOp?
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource* 
_output_shapes
:
??*
dtype02
MatMul/ReadVariableOpt
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
MatMul?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2	
BiasAddY
ReluReluBiasAdd:output:0*
T0*(
_output_shapes
:??????????2
Relu?
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
T0*(
_output_shapes
:??????????2

Identity"
identityIdentity:output:0*/
_input_shapes
:??????????::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:P L
(
_output_shapes
:??????????
 
_user_specified_nameinputs
?

*__inference_dense_106_layer_call_fn_675459

inputs
unknown
	unknown_0
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *N
fIRG
E__inference_dense_106_layer_call_and_return_conditional_losses_6747922
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*(
_output_shapes
:??????????2

Identity"
identityIdentity:output:0*/
_input_shapes
:??????????::22
StatefulPartitionedCallStatefulPartitionedCall:P L
(
_output_shapes
:??????????
 
_user_specified_nameinputs
?	
?
E__inference_dense_107_layer_call_and_return_conditional_losses_675470

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identity??BiasAdd/ReadVariableOp?MatMul/ReadVariableOp?
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource* 
_output_shapes
:
??*
dtype02
MatMul/ReadVariableOpt
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
MatMul?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2	
BiasAddY
ReluReluBiasAdd:output:0*
T0*(
_output_shapes
:??????????2
Relu?
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
T0*(
_output_shapes
:??????????2

Identity"
identityIdentity:output:0*/
_input_shapes
:??????????::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:P L
(
_output_shapes
:??????????
 
_user_specified_nameinputs
?	
?
B__inference_output_layer_call_and_return_conditional_losses_675570

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identity??BiasAdd/ReadVariableOp?MatMul/ReadVariableOp?
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	?+*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????+2
MatMul?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:+*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????+2	
BiasAdda
SoftmaxSoftmaxBiasAdd:output:0*
T0*'
_output_shapes
:?????????+2	
Softmax?
IdentityIdentitySoftmax:softmax:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
T0*'
_output_shapes
:?????????+2

Identity"
identityIdentity:output:0*/
_input_shapes
:??????????::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:P L
(
_output_shapes
:??????????
 
_user_specified_nameinputs
?	
?
E__inference_dense_105_layer_call_and_return_conditional_losses_674765

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identity??BiasAdd/ReadVariableOp?MatMul/ReadVariableOp?
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	F?*
dtype02
MatMul/ReadVariableOpt
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
MatMul?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2	
BiasAddY
ReluReluBiasAdd:output:0*
T0*(
_output_shapes
:??????????2
Relu?
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
T0*(
_output_shapes
:??????????2

Identity"
identityIdentity:output:0*.
_input_shapes
:?????????F::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:?????????F
 
_user_specified_nameinputs
?	
?
E__inference_dense_108_layer_call_and_return_conditional_losses_675490

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identity??BiasAdd/ReadVariableOp?MatMul/ReadVariableOp?
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource* 
_output_shapes
:
??*
dtype02
MatMul/ReadVariableOpt
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
MatMul?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2	
BiasAddY
ReluReluBiasAdd:output:0*
T0*(
_output_shapes
:??????????2
Relu?
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
T0*(
_output_shapes
:??????????2

Identity"
identityIdentity:output:0*/
_input_shapes
:??????????::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:P L
(
_output_shapes
:??????????
 
_user_specified_nameinputs
?L
?

D__inference_model_15_layer_call_and_return_conditional_losses_675345

inputs,
(dense_105_matmul_readvariableop_resource-
)dense_105_biasadd_readvariableop_resource,
(dense_106_matmul_readvariableop_resource-
)dense_106_biasadd_readvariableop_resource,
(dense_107_matmul_readvariableop_resource-
)dense_107_biasadd_readvariableop_resource,
(dense_108_matmul_readvariableop_resource-
)dense_108_biasadd_readvariableop_resource,
(dense_109_matmul_readvariableop_resource-
)dense_109_biasadd_readvariableop_resource,
(dense_110_matmul_readvariableop_resource-
)dense_110_biasadd_readvariableop_resource,
(dense_111_matmul_readvariableop_resource-
)dense_111_biasadd_readvariableop_resource)
%output_matmul_readvariableop_resource*
&output_biasadd_readvariableop_resource
identity?? dense_105/BiasAdd/ReadVariableOp?dense_105/MatMul/ReadVariableOp? dense_106/BiasAdd/ReadVariableOp?dense_106/MatMul/ReadVariableOp? dense_107/BiasAdd/ReadVariableOp?dense_107/MatMul/ReadVariableOp? dense_108/BiasAdd/ReadVariableOp?dense_108/MatMul/ReadVariableOp? dense_109/BiasAdd/ReadVariableOp?dense_109/MatMul/ReadVariableOp? dense_110/BiasAdd/ReadVariableOp?dense_110/MatMul/ReadVariableOp? dense_111/BiasAdd/ReadVariableOp?dense_111/MatMul/ReadVariableOp?output/BiasAdd/ReadVariableOp?output/MatMul/ReadVariableOp?
dense_105/MatMul/ReadVariableOpReadVariableOp(dense_105_matmul_readvariableop_resource*
_output_shapes
:	F?*
dtype02!
dense_105/MatMul/ReadVariableOp?
dense_105/MatMulMatMulinputs'dense_105/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
dense_105/MatMul?
 dense_105/BiasAdd/ReadVariableOpReadVariableOp)dense_105_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02"
 dense_105/BiasAdd/ReadVariableOp?
dense_105/BiasAddBiasAdddense_105/MatMul:product:0(dense_105/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
dense_105/BiasAddw
dense_105/ReluReludense_105/BiasAdd:output:0*
T0*(
_output_shapes
:??????????2
dense_105/Relu?
dense_106/MatMul/ReadVariableOpReadVariableOp(dense_106_matmul_readvariableop_resource* 
_output_shapes
:
??*
dtype02!
dense_106/MatMul/ReadVariableOp?
dense_106/MatMulMatMuldense_105/Relu:activations:0'dense_106/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
dense_106/MatMul?
 dense_106/BiasAdd/ReadVariableOpReadVariableOp)dense_106_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02"
 dense_106/BiasAdd/ReadVariableOp?
dense_106/BiasAddBiasAdddense_106/MatMul:product:0(dense_106/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
dense_106/BiasAddw
dense_106/ReluReludense_106/BiasAdd:output:0*
T0*(
_output_shapes
:??????????2
dense_106/Relu?
dense_107/MatMul/ReadVariableOpReadVariableOp(dense_107_matmul_readvariableop_resource* 
_output_shapes
:
??*
dtype02!
dense_107/MatMul/ReadVariableOp?
dense_107/MatMulMatMuldense_106/Relu:activations:0'dense_107/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
dense_107/MatMul?
 dense_107/BiasAdd/ReadVariableOpReadVariableOp)dense_107_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02"
 dense_107/BiasAdd/ReadVariableOp?
dense_107/BiasAddBiasAdddense_107/MatMul:product:0(dense_107/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
dense_107/BiasAddw
dense_107/ReluReludense_107/BiasAdd:output:0*
T0*(
_output_shapes
:??????????2
dense_107/Relu?
dense_108/MatMul/ReadVariableOpReadVariableOp(dense_108_matmul_readvariableop_resource* 
_output_shapes
:
??*
dtype02!
dense_108/MatMul/ReadVariableOp?
dense_108/MatMulMatMuldense_107/Relu:activations:0'dense_108/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
dense_108/MatMul?
 dense_108/BiasAdd/ReadVariableOpReadVariableOp)dense_108_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02"
 dense_108/BiasAdd/ReadVariableOp?
dense_108/BiasAddBiasAdddense_108/MatMul:product:0(dense_108/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
dense_108/BiasAddw
dense_108/ReluReludense_108/BiasAdd:output:0*
T0*(
_output_shapes
:??????????2
dense_108/Relu?
dense_109/MatMul/ReadVariableOpReadVariableOp(dense_109_matmul_readvariableop_resource* 
_output_shapes
:
??*
dtype02!
dense_109/MatMul/ReadVariableOp?
dense_109/MatMulMatMuldense_108/Relu:activations:0'dense_109/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
dense_109/MatMul?
 dense_109/BiasAdd/ReadVariableOpReadVariableOp)dense_109_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02"
 dense_109/BiasAdd/ReadVariableOp?
dense_109/BiasAddBiasAdddense_109/MatMul:product:0(dense_109/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
dense_109/BiasAddw
dense_109/ReluReludense_109/BiasAdd:output:0*
T0*(
_output_shapes
:??????????2
dense_109/Relu?
dense_110/MatMul/ReadVariableOpReadVariableOp(dense_110_matmul_readvariableop_resource* 
_output_shapes
:
??*
dtype02!
dense_110/MatMul/ReadVariableOp?
dense_110/MatMulMatMuldense_109/Relu:activations:0'dense_110/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
dense_110/MatMul?
 dense_110/BiasAdd/ReadVariableOpReadVariableOp)dense_110_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02"
 dense_110/BiasAdd/ReadVariableOp?
dense_110/BiasAddBiasAdddense_110/MatMul:product:0(dense_110/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
dense_110/BiasAddw
dense_110/ReluReludense_110/BiasAdd:output:0*
T0*(
_output_shapes
:??????????2
dense_110/Relu?
dense_111/MatMul/ReadVariableOpReadVariableOp(dense_111_matmul_readvariableop_resource* 
_output_shapes
:
??*
dtype02!
dense_111/MatMul/ReadVariableOp?
dense_111/MatMulMatMuldense_110/Relu:activations:0'dense_111/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
dense_111/MatMul?
 dense_111/BiasAdd/ReadVariableOpReadVariableOp)dense_111_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02"
 dense_111/BiasAdd/ReadVariableOp?
dense_111/BiasAddBiasAdddense_111/MatMul:product:0(dense_111/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
dense_111/BiasAddw
dense_111/ReluReludense_111/BiasAdd:output:0*
T0*(
_output_shapes
:??????????2
dense_111/Relu?
output/MatMul/ReadVariableOpReadVariableOp%output_matmul_readvariableop_resource*
_output_shapes
:	?+*
dtype02
output/MatMul/ReadVariableOp?
output/MatMulMatMuldense_111/Relu:activations:0$output/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????+2
output/MatMul?
output/BiasAdd/ReadVariableOpReadVariableOp&output_biasadd_readvariableop_resource*
_output_shapes
:+*
dtype02
output/BiasAdd/ReadVariableOp?
output/BiasAddBiasAddoutput/MatMul:product:0%output/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????+2
output/BiasAddv
output/SoftmaxSoftmaxoutput/BiasAdd:output:0*
T0*'
_output_shapes
:?????????+2
output/Softmax?
IdentityIdentityoutput/Softmax:softmax:0!^dense_105/BiasAdd/ReadVariableOp ^dense_105/MatMul/ReadVariableOp!^dense_106/BiasAdd/ReadVariableOp ^dense_106/MatMul/ReadVariableOp!^dense_107/BiasAdd/ReadVariableOp ^dense_107/MatMul/ReadVariableOp!^dense_108/BiasAdd/ReadVariableOp ^dense_108/MatMul/ReadVariableOp!^dense_109/BiasAdd/ReadVariableOp ^dense_109/MatMul/ReadVariableOp!^dense_110/BiasAdd/ReadVariableOp ^dense_110/MatMul/ReadVariableOp!^dense_111/BiasAdd/ReadVariableOp ^dense_111/MatMul/ReadVariableOp^output/BiasAdd/ReadVariableOp^output/MatMul/ReadVariableOp*
T0*'
_output_shapes
:?????????+2

Identity"
identityIdentity:output:0*f
_input_shapesU
S:?????????F::::::::::::::::2D
 dense_105/BiasAdd/ReadVariableOp dense_105/BiasAdd/ReadVariableOp2B
dense_105/MatMul/ReadVariableOpdense_105/MatMul/ReadVariableOp2D
 dense_106/BiasAdd/ReadVariableOp dense_106/BiasAdd/ReadVariableOp2B
dense_106/MatMul/ReadVariableOpdense_106/MatMul/ReadVariableOp2D
 dense_107/BiasAdd/ReadVariableOp dense_107/BiasAdd/ReadVariableOp2B
dense_107/MatMul/ReadVariableOpdense_107/MatMul/ReadVariableOp2D
 dense_108/BiasAdd/ReadVariableOp dense_108/BiasAdd/ReadVariableOp2B
dense_108/MatMul/ReadVariableOpdense_108/MatMul/ReadVariableOp2D
 dense_109/BiasAdd/ReadVariableOp dense_109/BiasAdd/ReadVariableOp2B
dense_109/MatMul/ReadVariableOpdense_109/MatMul/ReadVariableOp2D
 dense_110/BiasAdd/ReadVariableOp dense_110/BiasAdd/ReadVariableOp2B
dense_110/MatMul/ReadVariableOpdense_110/MatMul/ReadVariableOp2D
 dense_111/BiasAdd/ReadVariableOp dense_111/BiasAdd/ReadVariableOp2B
dense_111/MatMul/ReadVariableOpdense_111/MatMul/ReadVariableOp2>
output/BiasAdd/ReadVariableOpoutput/BiasAdd/ReadVariableOp2<
output/MatMul/ReadVariableOpoutput/MatMul/ReadVariableOp:O K
'
_output_shapes
:?????????F
 
_user_specified_nameinputs
?z
?
__inference__traced_save_675788
file_prefix/
+savev2_dense_105_kernel_read_readvariableop-
)savev2_dense_105_bias_read_readvariableop/
+savev2_dense_106_kernel_read_readvariableop-
)savev2_dense_106_bias_read_readvariableop/
+savev2_dense_107_kernel_read_readvariableop-
)savev2_dense_107_bias_read_readvariableop/
+savev2_dense_108_kernel_read_readvariableop-
)savev2_dense_108_bias_read_readvariableop/
+savev2_dense_109_kernel_read_readvariableop-
)savev2_dense_109_bias_read_readvariableop/
+savev2_dense_110_kernel_read_readvariableop-
)savev2_dense_110_bias_read_readvariableop/
+savev2_dense_111_kernel_read_readvariableop-
)savev2_dense_111_bias_read_readvariableop,
(savev2_output_kernel_read_readvariableop*
&savev2_output_bias_read_readvariableop)
%savev2_nadam_iter_read_readvariableop	+
'savev2_nadam_beta_1_read_readvariableop+
'savev2_nadam_beta_2_read_readvariableop*
&savev2_nadam_decay_read_readvariableop2
.savev2_nadam_learning_rate_read_readvariableop3
/savev2_nadam_momentum_cache_read_readvariableop$
 savev2_total_read_readvariableop$
 savev2_count_read_readvariableop&
"savev2_total_1_read_readvariableop&
"savev2_count_1_read_readvariableop*
&savev2_accumulator_read_readvariableop,
(savev2_accumulator_1_read_readvariableop,
(savev2_accumulator_2_read_readvariableop,
(savev2_accumulator_3_read_readvariableop7
3savev2_nadam_dense_105_kernel_m_read_readvariableop5
1savev2_nadam_dense_105_bias_m_read_readvariableop7
3savev2_nadam_dense_106_kernel_m_read_readvariableop5
1savev2_nadam_dense_106_bias_m_read_readvariableop7
3savev2_nadam_dense_107_kernel_m_read_readvariableop5
1savev2_nadam_dense_107_bias_m_read_readvariableop7
3savev2_nadam_dense_108_kernel_m_read_readvariableop5
1savev2_nadam_dense_108_bias_m_read_readvariableop7
3savev2_nadam_dense_109_kernel_m_read_readvariableop5
1savev2_nadam_dense_109_bias_m_read_readvariableop7
3savev2_nadam_dense_110_kernel_m_read_readvariableop5
1savev2_nadam_dense_110_bias_m_read_readvariableop7
3savev2_nadam_dense_111_kernel_m_read_readvariableop5
1savev2_nadam_dense_111_bias_m_read_readvariableop4
0savev2_nadam_output_kernel_m_read_readvariableop2
.savev2_nadam_output_bias_m_read_readvariableop7
3savev2_nadam_dense_105_kernel_v_read_readvariableop5
1savev2_nadam_dense_105_bias_v_read_readvariableop7
3savev2_nadam_dense_106_kernel_v_read_readvariableop5
1savev2_nadam_dense_106_bias_v_read_readvariableop7
3savev2_nadam_dense_107_kernel_v_read_readvariableop5
1savev2_nadam_dense_107_bias_v_read_readvariableop7
3savev2_nadam_dense_108_kernel_v_read_readvariableop5
1savev2_nadam_dense_108_bias_v_read_readvariableop7
3savev2_nadam_dense_109_kernel_v_read_readvariableop5
1savev2_nadam_dense_109_bias_v_read_readvariableop7
3savev2_nadam_dense_110_kernel_v_read_readvariableop5
1savev2_nadam_dense_110_bias_v_read_readvariableop7
3savev2_nadam_dense_111_kernel_v_read_readvariableop5
1savev2_nadam_dense_111_bias_v_read_readvariableop4
0savev2_nadam_output_kernel_v_read_readvariableop2
.savev2_nadam_output_bias_v_read_readvariableop
savev2_const

identity_1??MergeV2Checkpoints?
StaticRegexFullMatchStaticRegexFullMatchfile_prefix"/device:CPU:**
_output_shapes
: *
pattern
^s3://.*2
StaticRegexFullMatchc
ConstConst"/device:CPU:**
_output_shapes
: *
dtype0*
valueB B.part2
Constl
Const_1Const"/device:CPU:**
_output_shapes
: *
dtype0*
valueB B
_temp/part2	
Const_1?
SelectSelectStaticRegexFullMatch:output:0Const:output:0Const_1:output:0"/device:CPU:**
T0*
_output_shapes
: 2
Selectt

StringJoin
StringJoinfile_prefixSelect:output:0"/device:CPU:**
N*
_output_shapes
: 2

StringJoinZ

num_shardsConst*
_output_shapes
: *
dtype0*
value	B :2

num_shards
ShardedFilename/shardConst"/device:CPU:0*
_output_shapes
: *
dtype0*
value	B : 2
ShardedFilename/shard?
ShardedFilenameShardedFilenameStringJoin:output:0ShardedFilename/shard:output:0num_shards:output:0"/device:CPU:0*
_output_shapes
: 2
ShardedFilename?"
SaveV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:?*
dtype0*?!
value?!B?!?B6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-3/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-3/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-4/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-4/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-5/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-5/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-6/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-6/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-7/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-7/bias/.ATTRIBUTES/VARIABLE_VALUEB)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUEB*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUEB3optimizer/momentum_cache/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/count/.ATTRIBUTES/VARIABLE_VALUEB:keras_api/metrics/2/accumulator/.ATTRIBUTES/VARIABLE_VALUEB:keras_api/metrics/3/accumulator/.ATTRIBUTES/VARIABLE_VALUEB:keras_api/metrics/4/accumulator/.ATTRIBUTES/VARIABLE_VALUEB:keras_api/metrics/5/accumulator/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-3/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-3/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-4/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-4/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-5/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-5/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-6/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-6/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-7/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-7/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-3/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-3/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-4/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-4/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-5/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-5/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-6/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-6/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-7/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-7/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH2
SaveV2/tensor_names?
SaveV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:?*
dtype0*?
value?B??B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B 2
SaveV2/shape_and_slices?
SaveV2SaveV2ShardedFilename:filename:0SaveV2/tensor_names:output:0 SaveV2/shape_and_slices:output:0+savev2_dense_105_kernel_read_readvariableop)savev2_dense_105_bias_read_readvariableop+savev2_dense_106_kernel_read_readvariableop)savev2_dense_106_bias_read_readvariableop+savev2_dense_107_kernel_read_readvariableop)savev2_dense_107_bias_read_readvariableop+savev2_dense_108_kernel_read_readvariableop)savev2_dense_108_bias_read_readvariableop+savev2_dense_109_kernel_read_readvariableop)savev2_dense_109_bias_read_readvariableop+savev2_dense_110_kernel_read_readvariableop)savev2_dense_110_bias_read_readvariableop+savev2_dense_111_kernel_read_readvariableop)savev2_dense_111_bias_read_readvariableop(savev2_output_kernel_read_readvariableop&savev2_output_bias_read_readvariableop%savev2_nadam_iter_read_readvariableop'savev2_nadam_beta_1_read_readvariableop'savev2_nadam_beta_2_read_readvariableop&savev2_nadam_decay_read_readvariableop.savev2_nadam_learning_rate_read_readvariableop/savev2_nadam_momentum_cache_read_readvariableop savev2_total_read_readvariableop savev2_count_read_readvariableop"savev2_total_1_read_readvariableop"savev2_count_1_read_readvariableop&savev2_accumulator_read_readvariableop(savev2_accumulator_1_read_readvariableop(savev2_accumulator_2_read_readvariableop(savev2_accumulator_3_read_readvariableop3savev2_nadam_dense_105_kernel_m_read_readvariableop1savev2_nadam_dense_105_bias_m_read_readvariableop3savev2_nadam_dense_106_kernel_m_read_readvariableop1savev2_nadam_dense_106_bias_m_read_readvariableop3savev2_nadam_dense_107_kernel_m_read_readvariableop1savev2_nadam_dense_107_bias_m_read_readvariableop3savev2_nadam_dense_108_kernel_m_read_readvariableop1savev2_nadam_dense_108_bias_m_read_readvariableop3savev2_nadam_dense_109_kernel_m_read_readvariableop1savev2_nadam_dense_109_bias_m_read_readvariableop3savev2_nadam_dense_110_kernel_m_read_readvariableop1savev2_nadam_dense_110_bias_m_read_readvariableop3savev2_nadam_dense_111_kernel_m_read_readvariableop1savev2_nadam_dense_111_bias_m_read_readvariableop0savev2_nadam_output_kernel_m_read_readvariableop.savev2_nadam_output_bias_m_read_readvariableop3savev2_nadam_dense_105_kernel_v_read_readvariableop1savev2_nadam_dense_105_bias_v_read_readvariableop3savev2_nadam_dense_106_kernel_v_read_readvariableop1savev2_nadam_dense_106_bias_v_read_readvariableop3savev2_nadam_dense_107_kernel_v_read_readvariableop1savev2_nadam_dense_107_bias_v_read_readvariableop3savev2_nadam_dense_108_kernel_v_read_readvariableop1savev2_nadam_dense_108_bias_v_read_readvariableop3savev2_nadam_dense_109_kernel_v_read_readvariableop1savev2_nadam_dense_109_bias_v_read_readvariableop3savev2_nadam_dense_110_kernel_v_read_readvariableop1savev2_nadam_dense_110_bias_v_read_readvariableop3savev2_nadam_dense_111_kernel_v_read_readvariableop1savev2_nadam_dense_111_bias_v_read_readvariableop0savev2_nadam_output_kernel_v_read_readvariableop.savev2_nadam_output_bias_v_read_readvariableopsavev2_const"/device:CPU:0*
_output_shapes
 *M
dtypesC
A2?	2
SaveV2?
&MergeV2Checkpoints/checkpoint_prefixesPackShardedFilename:filename:0^SaveV2"/device:CPU:0*
N*
T0*
_output_shapes
:2(
&MergeV2Checkpoints/checkpoint_prefixes?
MergeV2CheckpointsMergeV2Checkpoints/MergeV2Checkpoints/checkpoint_prefixes:output:0file_prefix"/device:CPU:0*
_output_shapes
 2
MergeV2Checkpointsr
IdentityIdentityfile_prefix^MergeV2Checkpoints"/device:CPU:0*
T0*
_output_shapes
: 2

Identitym

Identity_1IdentityIdentity:output:0^MergeV2Checkpoints*
T0*
_output_shapes
: 2

Identity_1"!

identity_1Identity_1:output:0*?
_input_shapes?
?: :	F?:?:
??:?:
??:?:
??:?:
??:?:
??:?:
??:?:	?+:+: : : : : : : : : : :::::	F?:?:
??:?:
??:?:
??:?:
??:?:
??:?:
??:?:	?+:+:	F?:?:
??:?:
??:?:
??:?:
??:?:
??:?:
??:?:	?+:+: 2(
MergeV2CheckpointsMergeV2Checkpoints:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix:%!

_output_shapes
:	F?:!

_output_shapes	
:?:&"
 
_output_shapes
:
??:!

_output_shapes	
:?:&"
 
_output_shapes
:
??:!

_output_shapes	
:?:&"
 
_output_shapes
:
??:!

_output_shapes	
:?:&	"
 
_output_shapes
:
??:!


_output_shapes	
:?:&"
 
_output_shapes
:
??:!

_output_shapes	
:?:&"
 
_output_shapes
:
??:!

_output_shapes	
:?:%!

_output_shapes
:	?+: 

_output_shapes
:+:

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: : 

_output_shapes
:: 

_output_shapes
:: 

_output_shapes
:: 

_output_shapes
::%!

_output_shapes
:	F?:! 

_output_shapes	
:?:&!"
 
_output_shapes
:
??:!"

_output_shapes	
:?:&#"
 
_output_shapes
:
??:!$

_output_shapes	
:?:&%"
 
_output_shapes
:
??:!&

_output_shapes	
:?:&'"
 
_output_shapes
:
??:!(

_output_shapes	
:?:&)"
 
_output_shapes
:
??:!*

_output_shapes	
:?:&+"
 
_output_shapes
:
??:!,

_output_shapes	
:?:%-!

_output_shapes
:	?+: .

_output_shapes
:+:%/!

_output_shapes
:	F?:!0

_output_shapes	
:?:&1"
 
_output_shapes
:
??:!2

_output_shapes	
:?:&3"
 
_output_shapes
:
??:!4

_output_shapes	
:?:&5"
 
_output_shapes
:
??:!6

_output_shapes	
:?:&7"
 
_output_shapes
:
??:!8

_output_shapes	
:?:&9"
 
_output_shapes
:
??:!:

_output_shapes	
:?:&;"
 
_output_shapes
:
??:!<

_output_shapes	
:?:%=!

_output_shapes
:	?+: >

_output_shapes
:+:?

_output_shapes
: 
?

*__inference_dense_108_layer_call_fn_675499

inputs
unknown
	unknown_0
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *N
fIRG
E__inference_dense_108_layer_call_and_return_conditional_losses_6748462
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*(
_output_shapes
:??????????2

Identity"
identityIdentity:output:0*/
_input_shapes
:??????????::22
StatefulPartitionedCallStatefulPartitionedCall:P L
(
_output_shapes
:??????????
 
_user_specified_nameinputs
?	
?
B__inference_output_layer_call_and_return_conditional_losses_674954

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identity??BiasAdd/ReadVariableOp?MatMul/ReadVariableOp?
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	?+*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????+2
MatMul?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:+*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????+2	
BiasAdda
SoftmaxSoftmaxBiasAdd:output:0*
T0*'
_output_shapes
:?????????+2	
Softmax?
IdentityIdentitySoftmax:softmax:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
T0*'
_output_shapes
:?????????+2

Identity"
identityIdentity:output:0*/
_input_shapes
:??????????::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:P L
(
_output_shapes
:??????????
 
_user_specified_nameinputs
?*
?
D__inference_model_15_layer_call_and_return_conditional_losses_675015
first_input
dense_105_674974
dense_105_674976
dense_106_674979
dense_106_674981
dense_107_674984
dense_107_674986
dense_108_674989
dense_108_674991
dense_109_674994
dense_109_674996
dense_110_674999
dense_110_675001
dense_111_675004
dense_111_675006
output_675009
output_675011
identity??!dense_105/StatefulPartitionedCall?!dense_106/StatefulPartitionedCall?!dense_107/StatefulPartitionedCall?!dense_108/StatefulPartitionedCall?!dense_109/StatefulPartitionedCall?!dense_110/StatefulPartitionedCall?!dense_111/StatefulPartitionedCall?output/StatefulPartitionedCall?
!dense_105/StatefulPartitionedCallStatefulPartitionedCallfirst_inputdense_105_674974dense_105_674976*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *N
fIRG
E__inference_dense_105_layer_call_and_return_conditional_losses_6747652#
!dense_105/StatefulPartitionedCall?
!dense_106/StatefulPartitionedCallStatefulPartitionedCall*dense_105/StatefulPartitionedCall:output:0dense_106_674979dense_106_674981*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *N
fIRG
E__inference_dense_106_layer_call_and_return_conditional_losses_6747922#
!dense_106/StatefulPartitionedCall?
!dense_107/StatefulPartitionedCallStatefulPartitionedCall*dense_106/StatefulPartitionedCall:output:0dense_107_674984dense_107_674986*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *N
fIRG
E__inference_dense_107_layer_call_and_return_conditional_losses_6748192#
!dense_107/StatefulPartitionedCall?
!dense_108/StatefulPartitionedCallStatefulPartitionedCall*dense_107/StatefulPartitionedCall:output:0dense_108_674989dense_108_674991*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *N
fIRG
E__inference_dense_108_layer_call_and_return_conditional_losses_6748462#
!dense_108/StatefulPartitionedCall?
!dense_109/StatefulPartitionedCallStatefulPartitionedCall*dense_108/StatefulPartitionedCall:output:0dense_109_674994dense_109_674996*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *N
fIRG
E__inference_dense_109_layer_call_and_return_conditional_losses_6748732#
!dense_109/StatefulPartitionedCall?
!dense_110/StatefulPartitionedCallStatefulPartitionedCall*dense_109/StatefulPartitionedCall:output:0dense_110_674999dense_110_675001*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *N
fIRG
E__inference_dense_110_layer_call_and_return_conditional_losses_6749002#
!dense_110/StatefulPartitionedCall?
!dense_111/StatefulPartitionedCallStatefulPartitionedCall*dense_110/StatefulPartitionedCall:output:0dense_111_675004dense_111_675006*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *N
fIRG
E__inference_dense_111_layer_call_and_return_conditional_losses_6749272#
!dense_111/StatefulPartitionedCall?
output/StatefulPartitionedCallStatefulPartitionedCall*dense_111/StatefulPartitionedCall:output:0output_675009output_675011*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????+*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *K
fFRD
B__inference_output_layer_call_and_return_conditional_losses_6749542 
output/StatefulPartitionedCall?
IdentityIdentity'output/StatefulPartitionedCall:output:0"^dense_105/StatefulPartitionedCall"^dense_106/StatefulPartitionedCall"^dense_107/StatefulPartitionedCall"^dense_108/StatefulPartitionedCall"^dense_109/StatefulPartitionedCall"^dense_110/StatefulPartitionedCall"^dense_111/StatefulPartitionedCall^output/StatefulPartitionedCall*
T0*'
_output_shapes
:?????????+2

Identity"
identityIdentity:output:0*f
_input_shapesU
S:?????????F::::::::::::::::2F
!dense_105/StatefulPartitionedCall!dense_105/StatefulPartitionedCall2F
!dense_106/StatefulPartitionedCall!dense_106/StatefulPartitionedCall2F
!dense_107/StatefulPartitionedCall!dense_107/StatefulPartitionedCall2F
!dense_108/StatefulPartitionedCall!dense_108/StatefulPartitionedCall2F
!dense_109/StatefulPartitionedCall!dense_109/StatefulPartitionedCall2F
!dense_110/StatefulPartitionedCall!dense_110/StatefulPartitionedCall2F
!dense_111/StatefulPartitionedCall!dense_111/StatefulPartitionedCall2@
output/StatefulPartitionedCalloutput/StatefulPartitionedCall:T P
'
_output_shapes
:?????????F
%
_user_specified_namefirst_input
?	
?
E__inference_dense_109_layer_call_and_return_conditional_losses_674873

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identity??BiasAdd/ReadVariableOp?MatMul/ReadVariableOp?
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource* 
_output_shapes
:
??*
dtype02
MatMul/ReadVariableOpt
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
MatMul?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2	
BiasAddY
ReluReluBiasAdd:output:0*
T0*(
_output_shapes
:??????????2
Relu?
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
T0*(
_output_shapes
:??????????2

Identity"
identityIdentity:output:0*/
_input_shapes
:??????????::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:P L
(
_output_shapes
:??????????
 
_user_specified_nameinputs
?[
?
!__inference__wrapped_model_674750
first_input5
1model_15_dense_105_matmul_readvariableop_resource6
2model_15_dense_105_biasadd_readvariableop_resource5
1model_15_dense_106_matmul_readvariableop_resource6
2model_15_dense_106_biasadd_readvariableop_resource5
1model_15_dense_107_matmul_readvariableop_resource6
2model_15_dense_107_biasadd_readvariableop_resource5
1model_15_dense_108_matmul_readvariableop_resource6
2model_15_dense_108_biasadd_readvariableop_resource5
1model_15_dense_109_matmul_readvariableop_resource6
2model_15_dense_109_biasadd_readvariableop_resource5
1model_15_dense_110_matmul_readvariableop_resource6
2model_15_dense_110_biasadd_readvariableop_resource5
1model_15_dense_111_matmul_readvariableop_resource6
2model_15_dense_111_biasadd_readvariableop_resource2
.model_15_output_matmul_readvariableop_resource3
/model_15_output_biasadd_readvariableop_resource
identity??)model_15/dense_105/BiasAdd/ReadVariableOp?(model_15/dense_105/MatMul/ReadVariableOp?)model_15/dense_106/BiasAdd/ReadVariableOp?(model_15/dense_106/MatMul/ReadVariableOp?)model_15/dense_107/BiasAdd/ReadVariableOp?(model_15/dense_107/MatMul/ReadVariableOp?)model_15/dense_108/BiasAdd/ReadVariableOp?(model_15/dense_108/MatMul/ReadVariableOp?)model_15/dense_109/BiasAdd/ReadVariableOp?(model_15/dense_109/MatMul/ReadVariableOp?)model_15/dense_110/BiasAdd/ReadVariableOp?(model_15/dense_110/MatMul/ReadVariableOp?)model_15/dense_111/BiasAdd/ReadVariableOp?(model_15/dense_111/MatMul/ReadVariableOp?&model_15/output/BiasAdd/ReadVariableOp?%model_15/output/MatMul/ReadVariableOp?
(model_15/dense_105/MatMul/ReadVariableOpReadVariableOp1model_15_dense_105_matmul_readvariableop_resource*
_output_shapes
:	F?*
dtype02*
(model_15/dense_105/MatMul/ReadVariableOp?
model_15/dense_105/MatMulMatMulfirst_input0model_15/dense_105/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
model_15/dense_105/MatMul?
)model_15/dense_105/BiasAdd/ReadVariableOpReadVariableOp2model_15_dense_105_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02+
)model_15/dense_105/BiasAdd/ReadVariableOp?
model_15/dense_105/BiasAddBiasAdd#model_15/dense_105/MatMul:product:01model_15/dense_105/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
model_15/dense_105/BiasAdd?
model_15/dense_105/ReluRelu#model_15/dense_105/BiasAdd:output:0*
T0*(
_output_shapes
:??????????2
model_15/dense_105/Relu?
(model_15/dense_106/MatMul/ReadVariableOpReadVariableOp1model_15_dense_106_matmul_readvariableop_resource* 
_output_shapes
:
??*
dtype02*
(model_15/dense_106/MatMul/ReadVariableOp?
model_15/dense_106/MatMulMatMul%model_15/dense_105/Relu:activations:00model_15/dense_106/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
model_15/dense_106/MatMul?
)model_15/dense_106/BiasAdd/ReadVariableOpReadVariableOp2model_15_dense_106_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02+
)model_15/dense_106/BiasAdd/ReadVariableOp?
model_15/dense_106/BiasAddBiasAdd#model_15/dense_106/MatMul:product:01model_15/dense_106/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
model_15/dense_106/BiasAdd?
model_15/dense_106/ReluRelu#model_15/dense_106/BiasAdd:output:0*
T0*(
_output_shapes
:??????????2
model_15/dense_106/Relu?
(model_15/dense_107/MatMul/ReadVariableOpReadVariableOp1model_15_dense_107_matmul_readvariableop_resource* 
_output_shapes
:
??*
dtype02*
(model_15/dense_107/MatMul/ReadVariableOp?
model_15/dense_107/MatMulMatMul%model_15/dense_106/Relu:activations:00model_15/dense_107/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
model_15/dense_107/MatMul?
)model_15/dense_107/BiasAdd/ReadVariableOpReadVariableOp2model_15_dense_107_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02+
)model_15/dense_107/BiasAdd/ReadVariableOp?
model_15/dense_107/BiasAddBiasAdd#model_15/dense_107/MatMul:product:01model_15/dense_107/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
model_15/dense_107/BiasAdd?
model_15/dense_107/ReluRelu#model_15/dense_107/BiasAdd:output:0*
T0*(
_output_shapes
:??????????2
model_15/dense_107/Relu?
(model_15/dense_108/MatMul/ReadVariableOpReadVariableOp1model_15_dense_108_matmul_readvariableop_resource* 
_output_shapes
:
??*
dtype02*
(model_15/dense_108/MatMul/ReadVariableOp?
model_15/dense_108/MatMulMatMul%model_15/dense_107/Relu:activations:00model_15/dense_108/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
model_15/dense_108/MatMul?
)model_15/dense_108/BiasAdd/ReadVariableOpReadVariableOp2model_15_dense_108_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02+
)model_15/dense_108/BiasAdd/ReadVariableOp?
model_15/dense_108/BiasAddBiasAdd#model_15/dense_108/MatMul:product:01model_15/dense_108/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
model_15/dense_108/BiasAdd?
model_15/dense_108/ReluRelu#model_15/dense_108/BiasAdd:output:0*
T0*(
_output_shapes
:??????????2
model_15/dense_108/Relu?
(model_15/dense_109/MatMul/ReadVariableOpReadVariableOp1model_15_dense_109_matmul_readvariableop_resource* 
_output_shapes
:
??*
dtype02*
(model_15/dense_109/MatMul/ReadVariableOp?
model_15/dense_109/MatMulMatMul%model_15/dense_108/Relu:activations:00model_15/dense_109/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
model_15/dense_109/MatMul?
)model_15/dense_109/BiasAdd/ReadVariableOpReadVariableOp2model_15_dense_109_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02+
)model_15/dense_109/BiasAdd/ReadVariableOp?
model_15/dense_109/BiasAddBiasAdd#model_15/dense_109/MatMul:product:01model_15/dense_109/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
model_15/dense_109/BiasAdd?
model_15/dense_109/ReluRelu#model_15/dense_109/BiasAdd:output:0*
T0*(
_output_shapes
:??????????2
model_15/dense_109/Relu?
(model_15/dense_110/MatMul/ReadVariableOpReadVariableOp1model_15_dense_110_matmul_readvariableop_resource* 
_output_shapes
:
??*
dtype02*
(model_15/dense_110/MatMul/ReadVariableOp?
model_15/dense_110/MatMulMatMul%model_15/dense_109/Relu:activations:00model_15/dense_110/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
model_15/dense_110/MatMul?
)model_15/dense_110/BiasAdd/ReadVariableOpReadVariableOp2model_15_dense_110_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02+
)model_15/dense_110/BiasAdd/ReadVariableOp?
model_15/dense_110/BiasAddBiasAdd#model_15/dense_110/MatMul:product:01model_15/dense_110/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
model_15/dense_110/BiasAdd?
model_15/dense_110/ReluRelu#model_15/dense_110/BiasAdd:output:0*
T0*(
_output_shapes
:??????????2
model_15/dense_110/Relu?
(model_15/dense_111/MatMul/ReadVariableOpReadVariableOp1model_15_dense_111_matmul_readvariableop_resource* 
_output_shapes
:
??*
dtype02*
(model_15/dense_111/MatMul/ReadVariableOp?
model_15/dense_111/MatMulMatMul%model_15/dense_110/Relu:activations:00model_15/dense_111/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
model_15/dense_111/MatMul?
)model_15/dense_111/BiasAdd/ReadVariableOpReadVariableOp2model_15_dense_111_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02+
)model_15/dense_111/BiasAdd/ReadVariableOp?
model_15/dense_111/BiasAddBiasAdd#model_15/dense_111/MatMul:product:01model_15/dense_111/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
model_15/dense_111/BiasAdd?
model_15/dense_111/ReluRelu#model_15/dense_111/BiasAdd:output:0*
T0*(
_output_shapes
:??????????2
model_15/dense_111/Relu?
%model_15/output/MatMul/ReadVariableOpReadVariableOp.model_15_output_matmul_readvariableop_resource*
_output_shapes
:	?+*
dtype02'
%model_15/output/MatMul/ReadVariableOp?
model_15/output/MatMulMatMul%model_15/dense_111/Relu:activations:0-model_15/output/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????+2
model_15/output/MatMul?
&model_15/output/BiasAdd/ReadVariableOpReadVariableOp/model_15_output_biasadd_readvariableop_resource*
_output_shapes
:+*
dtype02(
&model_15/output/BiasAdd/ReadVariableOp?
model_15/output/BiasAddBiasAdd model_15/output/MatMul:product:0.model_15/output/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????+2
model_15/output/BiasAdd?
model_15/output/SoftmaxSoftmax model_15/output/BiasAdd:output:0*
T0*'
_output_shapes
:?????????+2
model_15/output/Softmax?
IdentityIdentity!model_15/output/Softmax:softmax:0*^model_15/dense_105/BiasAdd/ReadVariableOp)^model_15/dense_105/MatMul/ReadVariableOp*^model_15/dense_106/BiasAdd/ReadVariableOp)^model_15/dense_106/MatMul/ReadVariableOp*^model_15/dense_107/BiasAdd/ReadVariableOp)^model_15/dense_107/MatMul/ReadVariableOp*^model_15/dense_108/BiasAdd/ReadVariableOp)^model_15/dense_108/MatMul/ReadVariableOp*^model_15/dense_109/BiasAdd/ReadVariableOp)^model_15/dense_109/MatMul/ReadVariableOp*^model_15/dense_110/BiasAdd/ReadVariableOp)^model_15/dense_110/MatMul/ReadVariableOp*^model_15/dense_111/BiasAdd/ReadVariableOp)^model_15/dense_111/MatMul/ReadVariableOp'^model_15/output/BiasAdd/ReadVariableOp&^model_15/output/MatMul/ReadVariableOp*
T0*'
_output_shapes
:?????????+2

Identity"
identityIdentity:output:0*f
_input_shapesU
S:?????????F::::::::::::::::2V
)model_15/dense_105/BiasAdd/ReadVariableOp)model_15/dense_105/BiasAdd/ReadVariableOp2T
(model_15/dense_105/MatMul/ReadVariableOp(model_15/dense_105/MatMul/ReadVariableOp2V
)model_15/dense_106/BiasAdd/ReadVariableOp)model_15/dense_106/BiasAdd/ReadVariableOp2T
(model_15/dense_106/MatMul/ReadVariableOp(model_15/dense_106/MatMul/ReadVariableOp2V
)model_15/dense_107/BiasAdd/ReadVariableOp)model_15/dense_107/BiasAdd/ReadVariableOp2T
(model_15/dense_107/MatMul/ReadVariableOp(model_15/dense_107/MatMul/ReadVariableOp2V
)model_15/dense_108/BiasAdd/ReadVariableOp)model_15/dense_108/BiasAdd/ReadVariableOp2T
(model_15/dense_108/MatMul/ReadVariableOp(model_15/dense_108/MatMul/ReadVariableOp2V
)model_15/dense_109/BiasAdd/ReadVariableOp)model_15/dense_109/BiasAdd/ReadVariableOp2T
(model_15/dense_109/MatMul/ReadVariableOp(model_15/dense_109/MatMul/ReadVariableOp2V
)model_15/dense_110/BiasAdd/ReadVariableOp)model_15/dense_110/BiasAdd/ReadVariableOp2T
(model_15/dense_110/MatMul/ReadVariableOp(model_15/dense_110/MatMul/ReadVariableOp2V
)model_15/dense_111/BiasAdd/ReadVariableOp)model_15/dense_111/BiasAdd/ReadVariableOp2T
(model_15/dense_111/MatMul/ReadVariableOp(model_15/dense_111/MatMul/ReadVariableOp2P
&model_15/output/BiasAdd/ReadVariableOp&model_15/output/BiasAdd/ReadVariableOp2N
%model_15/output/MatMul/ReadVariableOp%model_15/output/MatMul/ReadVariableOp:T P
'
_output_shapes
:?????????F
%
_user_specified_namefirst_input
?

*__inference_dense_105_layer_call_fn_675439

inputs
unknown
	unknown_0
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *N
fIRG
E__inference_dense_105_layer_call_and_return_conditional_losses_6747652
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*(
_output_shapes
:??????????2

Identity"
identityIdentity:output:0*.
_input_shapes
:?????????F::22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:?????????F
 
_user_specified_nameinputs
?

?
)__inference_model_15_layer_call_fn_675419

inputs
unknown
	unknown_0
	unknown_1
	unknown_2
	unknown_3
	unknown_4
	unknown_5
	unknown_6
	unknown_7
	unknown_8
	unknown_9

unknown_10

unknown_11

unknown_12

unknown_13

unknown_14
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13
unknown_14*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????+*2
_read_only_resource_inputs
	
*-
config_proto

CPU

GPU 2J 8? *M
fHRF
D__inference_model_15_layer_call_and_return_conditional_losses_6751432
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:?????????+2

Identity"
identityIdentity:output:0*f
_input_shapesU
S:?????????F::::::::::::::::22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:?????????F
 
_user_specified_nameinputs
?	
?
E__inference_dense_110_layer_call_and_return_conditional_losses_674900

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identity??BiasAdd/ReadVariableOp?MatMul/ReadVariableOp?
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource* 
_output_shapes
:
??*
dtype02
MatMul/ReadVariableOpt
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
MatMul?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2	
BiasAddY
ReluReluBiasAdd:output:0*
T0*(
_output_shapes
:??????????2
Relu?
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
T0*(
_output_shapes
:??????????2

Identity"
identityIdentity:output:0*/
_input_shapes
:??????????::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:P L
(
_output_shapes
:??????????
 
_user_specified_nameinputs
?*
?
D__inference_model_15_layer_call_and_return_conditional_losses_675143

inputs
dense_105_675102
dense_105_675104
dense_106_675107
dense_106_675109
dense_107_675112
dense_107_675114
dense_108_675117
dense_108_675119
dense_109_675122
dense_109_675124
dense_110_675127
dense_110_675129
dense_111_675132
dense_111_675134
output_675137
output_675139
identity??!dense_105/StatefulPartitionedCall?!dense_106/StatefulPartitionedCall?!dense_107/StatefulPartitionedCall?!dense_108/StatefulPartitionedCall?!dense_109/StatefulPartitionedCall?!dense_110/StatefulPartitionedCall?!dense_111/StatefulPartitionedCall?output/StatefulPartitionedCall?
!dense_105/StatefulPartitionedCallStatefulPartitionedCallinputsdense_105_675102dense_105_675104*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *N
fIRG
E__inference_dense_105_layer_call_and_return_conditional_losses_6747652#
!dense_105/StatefulPartitionedCall?
!dense_106/StatefulPartitionedCallStatefulPartitionedCall*dense_105/StatefulPartitionedCall:output:0dense_106_675107dense_106_675109*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *N
fIRG
E__inference_dense_106_layer_call_and_return_conditional_losses_6747922#
!dense_106/StatefulPartitionedCall?
!dense_107/StatefulPartitionedCallStatefulPartitionedCall*dense_106/StatefulPartitionedCall:output:0dense_107_675112dense_107_675114*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *N
fIRG
E__inference_dense_107_layer_call_and_return_conditional_losses_6748192#
!dense_107/StatefulPartitionedCall?
!dense_108/StatefulPartitionedCallStatefulPartitionedCall*dense_107/StatefulPartitionedCall:output:0dense_108_675117dense_108_675119*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *N
fIRG
E__inference_dense_108_layer_call_and_return_conditional_losses_6748462#
!dense_108/StatefulPartitionedCall?
!dense_109/StatefulPartitionedCallStatefulPartitionedCall*dense_108/StatefulPartitionedCall:output:0dense_109_675122dense_109_675124*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *N
fIRG
E__inference_dense_109_layer_call_and_return_conditional_losses_6748732#
!dense_109/StatefulPartitionedCall?
!dense_110/StatefulPartitionedCallStatefulPartitionedCall*dense_109/StatefulPartitionedCall:output:0dense_110_675127dense_110_675129*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *N
fIRG
E__inference_dense_110_layer_call_and_return_conditional_losses_6749002#
!dense_110/StatefulPartitionedCall?
!dense_111/StatefulPartitionedCallStatefulPartitionedCall*dense_110/StatefulPartitionedCall:output:0dense_111_675132dense_111_675134*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *N
fIRG
E__inference_dense_111_layer_call_and_return_conditional_losses_6749272#
!dense_111/StatefulPartitionedCall?
output/StatefulPartitionedCallStatefulPartitionedCall*dense_111/StatefulPartitionedCall:output:0output_675137output_675139*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????+*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *K
fFRD
B__inference_output_layer_call_and_return_conditional_losses_6749542 
output/StatefulPartitionedCall?
IdentityIdentity'output/StatefulPartitionedCall:output:0"^dense_105/StatefulPartitionedCall"^dense_106/StatefulPartitionedCall"^dense_107/StatefulPartitionedCall"^dense_108/StatefulPartitionedCall"^dense_109/StatefulPartitionedCall"^dense_110/StatefulPartitionedCall"^dense_111/StatefulPartitionedCall^output/StatefulPartitionedCall*
T0*'
_output_shapes
:?????????+2

Identity"
identityIdentity:output:0*f
_input_shapesU
S:?????????F::::::::::::::::2F
!dense_105/StatefulPartitionedCall!dense_105/StatefulPartitionedCall2F
!dense_106/StatefulPartitionedCall!dense_106/StatefulPartitionedCall2F
!dense_107/StatefulPartitionedCall!dense_107/StatefulPartitionedCall2F
!dense_108/StatefulPartitionedCall!dense_108/StatefulPartitionedCall2F
!dense_109/StatefulPartitionedCall!dense_109/StatefulPartitionedCall2F
!dense_110/StatefulPartitionedCall!dense_110/StatefulPartitionedCall2F
!dense_111/StatefulPartitionedCall!dense_111/StatefulPartitionedCall2@
output/StatefulPartitionedCalloutput/StatefulPartitionedCall:O K
'
_output_shapes
:?????????F
 
_user_specified_nameinputs
?

*__inference_dense_111_layer_call_fn_675559

inputs
unknown
	unknown_0
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *N
fIRG
E__inference_dense_111_layer_call_and_return_conditional_losses_6749272
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*(
_output_shapes
:??????????2

Identity"
identityIdentity:output:0*/
_input_shapes
:??????????::22
StatefulPartitionedCallStatefulPartitionedCall:P L
(
_output_shapes
:??????????
 
_user_specified_nameinputs
?	
?
E__inference_dense_106_layer_call_and_return_conditional_losses_675450

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identity??BiasAdd/ReadVariableOp?MatMul/ReadVariableOp?
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource* 
_output_shapes
:
??*
dtype02
MatMul/ReadVariableOpt
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
MatMul?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2	
BiasAddY
ReluReluBiasAdd:output:0*
T0*(
_output_shapes
:??????????2
Relu?
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
T0*(
_output_shapes
:??????????2

Identity"
identityIdentity:output:0*/
_input_shapes
:??????????::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:P L
(
_output_shapes
:??????????
 
_user_specified_nameinputs
?	
?
E__inference_dense_105_layer_call_and_return_conditional_losses_675430

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identity??BiasAdd/ReadVariableOp?MatMul/ReadVariableOp?
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	F?*
dtype02
MatMul/ReadVariableOpt
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
MatMul?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2	
BiasAddY
ReluReluBiasAdd:output:0*
T0*(
_output_shapes
:??????????2
Relu?
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
T0*(
_output_shapes
:??????????2

Identity"
identityIdentity:output:0*.
_input_shapes
:?????????F::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:?????????F
 
_user_specified_nameinputs
?	
?
E__inference_dense_111_layer_call_and_return_conditional_losses_675550

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identity??BiasAdd/ReadVariableOp?MatMul/ReadVariableOp?
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource* 
_output_shapes
:
??*
dtype02
MatMul/ReadVariableOpt
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
MatMul?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2	
BiasAddY
ReluReluBiasAdd:output:0*
T0*(
_output_shapes
:??????????2
Relu?
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
T0*(
_output_shapes
:??????????2

Identity"
identityIdentity:output:0*/
_input_shapes
:??????????::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:P L
(
_output_shapes
:??????????
 
_user_specified_nameinputs
?	
?
E__inference_dense_111_layer_call_and_return_conditional_losses_674927

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identity??BiasAdd/ReadVariableOp?MatMul/ReadVariableOp?
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource* 
_output_shapes
:
??*
dtype02
MatMul/ReadVariableOpt
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
MatMul?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2	
BiasAddY
ReluReluBiasAdd:output:0*
T0*(
_output_shapes
:??????????2
Relu?
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
T0*(
_output_shapes
:??????????2

Identity"
identityIdentity:output:0*/
_input_shapes
:??????????::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:P L
(
_output_shapes
:??????????
 
_user_specified_nameinputs
ڃ
? 
"__inference__traced_restore_675984
file_prefix%
!assignvariableop_dense_105_kernel%
!assignvariableop_1_dense_105_bias'
#assignvariableop_2_dense_106_kernel%
!assignvariableop_3_dense_106_bias'
#assignvariableop_4_dense_107_kernel%
!assignvariableop_5_dense_107_bias'
#assignvariableop_6_dense_108_kernel%
!assignvariableop_7_dense_108_bias'
#assignvariableop_8_dense_109_kernel%
!assignvariableop_9_dense_109_bias(
$assignvariableop_10_dense_110_kernel&
"assignvariableop_11_dense_110_bias(
$assignvariableop_12_dense_111_kernel&
"assignvariableop_13_dense_111_bias%
!assignvariableop_14_output_kernel#
assignvariableop_15_output_bias"
assignvariableop_16_nadam_iter$
 assignvariableop_17_nadam_beta_1$
 assignvariableop_18_nadam_beta_2#
assignvariableop_19_nadam_decay+
'assignvariableop_20_nadam_learning_rate,
(assignvariableop_21_nadam_momentum_cache
assignvariableop_22_total
assignvariableop_23_count
assignvariableop_24_total_1
assignvariableop_25_count_1#
assignvariableop_26_accumulator%
!assignvariableop_27_accumulator_1%
!assignvariableop_28_accumulator_2%
!assignvariableop_29_accumulator_30
,assignvariableop_30_nadam_dense_105_kernel_m.
*assignvariableop_31_nadam_dense_105_bias_m0
,assignvariableop_32_nadam_dense_106_kernel_m.
*assignvariableop_33_nadam_dense_106_bias_m0
,assignvariableop_34_nadam_dense_107_kernel_m.
*assignvariableop_35_nadam_dense_107_bias_m0
,assignvariableop_36_nadam_dense_108_kernel_m.
*assignvariableop_37_nadam_dense_108_bias_m0
,assignvariableop_38_nadam_dense_109_kernel_m.
*assignvariableop_39_nadam_dense_109_bias_m0
,assignvariableop_40_nadam_dense_110_kernel_m.
*assignvariableop_41_nadam_dense_110_bias_m0
,assignvariableop_42_nadam_dense_111_kernel_m.
*assignvariableop_43_nadam_dense_111_bias_m-
)assignvariableop_44_nadam_output_kernel_m+
'assignvariableop_45_nadam_output_bias_m0
,assignvariableop_46_nadam_dense_105_kernel_v.
*assignvariableop_47_nadam_dense_105_bias_v0
,assignvariableop_48_nadam_dense_106_kernel_v.
*assignvariableop_49_nadam_dense_106_bias_v0
,assignvariableop_50_nadam_dense_107_kernel_v.
*assignvariableop_51_nadam_dense_107_bias_v0
,assignvariableop_52_nadam_dense_108_kernel_v.
*assignvariableop_53_nadam_dense_108_bias_v0
,assignvariableop_54_nadam_dense_109_kernel_v.
*assignvariableop_55_nadam_dense_109_bias_v0
,assignvariableop_56_nadam_dense_110_kernel_v.
*assignvariableop_57_nadam_dense_110_bias_v0
,assignvariableop_58_nadam_dense_111_kernel_v.
*assignvariableop_59_nadam_dense_111_bias_v-
)assignvariableop_60_nadam_output_kernel_v+
'assignvariableop_61_nadam_output_bias_v
identity_63??AssignVariableOp?AssignVariableOp_1?AssignVariableOp_10?AssignVariableOp_11?AssignVariableOp_12?AssignVariableOp_13?AssignVariableOp_14?AssignVariableOp_15?AssignVariableOp_16?AssignVariableOp_17?AssignVariableOp_18?AssignVariableOp_19?AssignVariableOp_2?AssignVariableOp_20?AssignVariableOp_21?AssignVariableOp_22?AssignVariableOp_23?AssignVariableOp_24?AssignVariableOp_25?AssignVariableOp_26?AssignVariableOp_27?AssignVariableOp_28?AssignVariableOp_29?AssignVariableOp_3?AssignVariableOp_30?AssignVariableOp_31?AssignVariableOp_32?AssignVariableOp_33?AssignVariableOp_34?AssignVariableOp_35?AssignVariableOp_36?AssignVariableOp_37?AssignVariableOp_38?AssignVariableOp_39?AssignVariableOp_4?AssignVariableOp_40?AssignVariableOp_41?AssignVariableOp_42?AssignVariableOp_43?AssignVariableOp_44?AssignVariableOp_45?AssignVariableOp_46?AssignVariableOp_47?AssignVariableOp_48?AssignVariableOp_49?AssignVariableOp_5?AssignVariableOp_50?AssignVariableOp_51?AssignVariableOp_52?AssignVariableOp_53?AssignVariableOp_54?AssignVariableOp_55?AssignVariableOp_56?AssignVariableOp_57?AssignVariableOp_58?AssignVariableOp_59?AssignVariableOp_6?AssignVariableOp_60?AssignVariableOp_61?AssignVariableOp_7?AssignVariableOp_8?AssignVariableOp_9?"
RestoreV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:?*
dtype0*?!
value?!B?!?B6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-3/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-3/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-4/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-4/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-5/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-5/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-6/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-6/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-7/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-7/bias/.ATTRIBUTES/VARIABLE_VALUEB)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUEB*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUEB3optimizer/momentum_cache/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/count/.ATTRIBUTES/VARIABLE_VALUEB:keras_api/metrics/2/accumulator/.ATTRIBUTES/VARIABLE_VALUEB:keras_api/metrics/3/accumulator/.ATTRIBUTES/VARIABLE_VALUEB:keras_api/metrics/4/accumulator/.ATTRIBUTES/VARIABLE_VALUEB:keras_api/metrics/5/accumulator/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-3/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-3/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-4/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-4/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-5/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-5/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-6/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-6/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-7/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-7/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-3/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-3/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-4/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-4/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-5/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-5/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-6/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-6/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-7/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-7/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH2
RestoreV2/tensor_names?
RestoreV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:?*
dtype0*?
value?B??B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B 2
RestoreV2/shape_and_slices?
	RestoreV2	RestoreV2file_prefixRestoreV2/tensor_names:output:0#RestoreV2/shape_and_slices:output:0"/device:CPU:0*?
_output_shapes?
?:::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::*M
dtypesC
A2?	2
	RestoreV2g
IdentityIdentityRestoreV2:tensors:0"/device:CPU:0*
T0*
_output_shapes
:2

Identity?
AssignVariableOpAssignVariableOp!assignvariableop_dense_105_kernelIdentity:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOpk

Identity_1IdentityRestoreV2:tensors:1"/device:CPU:0*
T0*
_output_shapes
:2

Identity_1?
AssignVariableOp_1AssignVariableOp!assignvariableop_1_dense_105_biasIdentity_1:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_1k

Identity_2IdentityRestoreV2:tensors:2"/device:CPU:0*
T0*
_output_shapes
:2

Identity_2?
AssignVariableOp_2AssignVariableOp#assignvariableop_2_dense_106_kernelIdentity_2:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_2k

Identity_3IdentityRestoreV2:tensors:3"/device:CPU:0*
T0*
_output_shapes
:2

Identity_3?
AssignVariableOp_3AssignVariableOp!assignvariableop_3_dense_106_biasIdentity_3:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_3k

Identity_4IdentityRestoreV2:tensors:4"/device:CPU:0*
T0*
_output_shapes
:2

Identity_4?
AssignVariableOp_4AssignVariableOp#assignvariableop_4_dense_107_kernelIdentity_4:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_4k

Identity_5IdentityRestoreV2:tensors:5"/device:CPU:0*
T0*
_output_shapes
:2

Identity_5?
AssignVariableOp_5AssignVariableOp!assignvariableop_5_dense_107_biasIdentity_5:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_5k

Identity_6IdentityRestoreV2:tensors:6"/device:CPU:0*
T0*
_output_shapes
:2

Identity_6?
AssignVariableOp_6AssignVariableOp#assignvariableop_6_dense_108_kernelIdentity_6:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_6k

Identity_7IdentityRestoreV2:tensors:7"/device:CPU:0*
T0*
_output_shapes
:2

Identity_7?
AssignVariableOp_7AssignVariableOp!assignvariableop_7_dense_108_biasIdentity_7:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_7k

Identity_8IdentityRestoreV2:tensors:8"/device:CPU:0*
T0*
_output_shapes
:2

Identity_8?
AssignVariableOp_8AssignVariableOp#assignvariableop_8_dense_109_kernelIdentity_8:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_8k

Identity_9IdentityRestoreV2:tensors:9"/device:CPU:0*
T0*
_output_shapes
:2

Identity_9?
AssignVariableOp_9AssignVariableOp!assignvariableop_9_dense_109_biasIdentity_9:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_9n
Identity_10IdentityRestoreV2:tensors:10"/device:CPU:0*
T0*
_output_shapes
:2
Identity_10?
AssignVariableOp_10AssignVariableOp$assignvariableop_10_dense_110_kernelIdentity_10:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_10n
Identity_11IdentityRestoreV2:tensors:11"/device:CPU:0*
T0*
_output_shapes
:2
Identity_11?
AssignVariableOp_11AssignVariableOp"assignvariableop_11_dense_110_biasIdentity_11:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_11n
Identity_12IdentityRestoreV2:tensors:12"/device:CPU:0*
T0*
_output_shapes
:2
Identity_12?
AssignVariableOp_12AssignVariableOp$assignvariableop_12_dense_111_kernelIdentity_12:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_12n
Identity_13IdentityRestoreV2:tensors:13"/device:CPU:0*
T0*
_output_shapes
:2
Identity_13?
AssignVariableOp_13AssignVariableOp"assignvariableop_13_dense_111_biasIdentity_13:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_13n
Identity_14IdentityRestoreV2:tensors:14"/device:CPU:0*
T0*
_output_shapes
:2
Identity_14?
AssignVariableOp_14AssignVariableOp!assignvariableop_14_output_kernelIdentity_14:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_14n
Identity_15IdentityRestoreV2:tensors:15"/device:CPU:0*
T0*
_output_shapes
:2
Identity_15?
AssignVariableOp_15AssignVariableOpassignvariableop_15_output_biasIdentity_15:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_15n
Identity_16IdentityRestoreV2:tensors:16"/device:CPU:0*
T0	*
_output_shapes
:2
Identity_16?
AssignVariableOp_16AssignVariableOpassignvariableop_16_nadam_iterIdentity_16:output:0"/device:CPU:0*
_output_shapes
 *
dtype0	2
AssignVariableOp_16n
Identity_17IdentityRestoreV2:tensors:17"/device:CPU:0*
T0*
_output_shapes
:2
Identity_17?
AssignVariableOp_17AssignVariableOp assignvariableop_17_nadam_beta_1Identity_17:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_17n
Identity_18IdentityRestoreV2:tensors:18"/device:CPU:0*
T0*
_output_shapes
:2
Identity_18?
AssignVariableOp_18AssignVariableOp assignvariableop_18_nadam_beta_2Identity_18:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_18n
Identity_19IdentityRestoreV2:tensors:19"/device:CPU:0*
T0*
_output_shapes
:2
Identity_19?
AssignVariableOp_19AssignVariableOpassignvariableop_19_nadam_decayIdentity_19:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_19n
Identity_20IdentityRestoreV2:tensors:20"/device:CPU:0*
T0*
_output_shapes
:2
Identity_20?
AssignVariableOp_20AssignVariableOp'assignvariableop_20_nadam_learning_rateIdentity_20:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_20n
Identity_21IdentityRestoreV2:tensors:21"/device:CPU:0*
T0*
_output_shapes
:2
Identity_21?
AssignVariableOp_21AssignVariableOp(assignvariableop_21_nadam_momentum_cacheIdentity_21:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_21n
Identity_22IdentityRestoreV2:tensors:22"/device:CPU:0*
T0*
_output_shapes
:2
Identity_22?
AssignVariableOp_22AssignVariableOpassignvariableop_22_totalIdentity_22:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_22n
Identity_23IdentityRestoreV2:tensors:23"/device:CPU:0*
T0*
_output_shapes
:2
Identity_23?
AssignVariableOp_23AssignVariableOpassignvariableop_23_countIdentity_23:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_23n
Identity_24IdentityRestoreV2:tensors:24"/device:CPU:0*
T0*
_output_shapes
:2
Identity_24?
AssignVariableOp_24AssignVariableOpassignvariableop_24_total_1Identity_24:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_24n
Identity_25IdentityRestoreV2:tensors:25"/device:CPU:0*
T0*
_output_shapes
:2
Identity_25?
AssignVariableOp_25AssignVariableOpassignvariableop_25_count_1Identity_25:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_25n
Identity_26IdentityRestoreV2:tensors:26"/device:CPU:0*
T0*
_output_shapes
:2
Identity_26?
AssignVariableOp_26AssignVariableOpassignvariableop_26_accumulatorIdentity_26:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_26n
Identity_27IdentityRestoreV2:tensors:27"/device:CPU:0*
T0*
_output_shapes
:2
Identity_27?
AssignVariableOp_27AssignVariableOp!assignvariableop_27_accumulator_1Identity_27:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_27n
Identity_28IdentityRestoreV2:tensors:28"/device:CPU:0*
T0*
_output_shapes
:2
Identity_28?
AssignVariableOp_28AssignVariableOp!assignvariableop_28_accumulator_2Identity_28:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_28n
Identity_29IdentityRestoreV2:tensors:29"/device:CPU:0*
T0*
_output_shapes
:2
Identity_29?
AssignVariableOp_29AssignVariableOp!assignvariableop_29_accumulator_3Identity_29:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_29n
Identity_30IdentityRestoreV2:tensors:30"/device:CPU:0*
T0*
_output_shapes
:2
Identity_30?
AssignVariableOp_30AssignVariableOp,assignvariableop_30_nadam_dense_105_kernel_mIdentity_30:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_30n
Identity_31IdentityRestoreV2:tensors:31"/device:CPU:0*
T0*
_output_shapes
:2
Identity_31?
AssignVariableOp_31AssignVariableOp*assignvariableop_31_nadam_dense_105_bias_mIdentity_31:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_31n
Identity_32IdentityRestoreV2:tensors:32"/device:CPU:0*
T0*
_output_shapes
:2
Identity_32?
AssignVariableOp_32AssignVariableOp,assignvariableop_32_nadam_dense_106_kernel_mIdentity_32:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_32n
Identity_33IdentityRestoreV2:tensors:33"/device:CPU:0*
T0*
_output_shapes
:2
Identity_33?
AssignVariableOp_33AssignVariableOp*assignvariableop_33_nadam_dense_106_bias_mIdentity_33:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_33n
Identity_34IdentityRestoreV2:tensors:34"/device:CPU:0*
T0*
_output_shapes
:2
Identity_34?
AssignVariableOp_34AssignVariableOp,assignvariableop_34_nadam_dense_107_kernel_mIdentity_34:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_34n
Identity_35IdentityRestoreV2:tensors:35"/device:CPU:0*
T0*
_output_shapes
:2
Identity_35?
AssignVariableOp_35AssignVariableOp*assignvariableop_35_nadam_dense_107_bias_mIdentity_35:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_35n
Identity_36IdentityRestoreV2:tensors:36"/device:CPU:0*
T0*
_output_shapes
:2
Identity_36?
AssignVariableOp_36AssignVariableOp,assignvariableop_36_nadam_dense_108_kernel_mIdentity_36:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_36n
Identity_37IdentityRestoreV2:tensors:37"/device:CPU:0*
T0*
_output_shapes
:2
Identity_37?
AssignVariableOp_37AssignVariableOp*assignvariableop_37_nadam_dense_108_bias_mIdentity_37:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_37n
Identity_38IdentityRestoreV2:tensors:38"/device:CPU:0*
T0*
_output_shapes
:2
Identity_38?
AssignVariableOp_38AssignVariableOp,assignvariableop_38_nadam_dense_109_kernel_mIdentity_38:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_38n
Identity_39IdentityRestoreV2:tensors:39"/device:CPU:0*
T0*
_output_shapes
:2
Identity_39?
AssignVariableOp_39AssignVariableOp*assignvariableop_39_nadam_dense_109_bias_mIdentity_39:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_39n
Identity_40IdentityRestoreV2:tensors:40"/device:CPU:0*
T0*
_output_shapes
:2
Identity_40?
AssignVariableOp_40AssignVariableOp,assignvariableop_40_nadam_dense_110_kernel_mIdentity_40:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_40n
Identity_41IdentityRestoreV2:tensors:41"/device:CPU:0*
T0*
_output_shapes
:2
Identity_41?
AssignVariableOp_41AssignVariableOp*assignvariableop_41_nadam_dense_110_bias_mIdentity_41:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_41n
Identity_42IdentityRestoreV2:tensors:42"/device:CPU:0*
T0*
_output_shapes
:2
Identity_42?
AssignVariableOp_42AssignVariableOp,assignvariableop_42_nadam_dense_111_kernel_mIdentity_42:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_42n
Identity_43IdentityRestoreV2:tensors:43"/device:CPU:0*
T0*
_output_shapes
:2
Identity_43?
AssignVariableOp_43AssignVariableOp*assignvariableop_43_nadam_dense_111_bias_mIdentity_43:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_43n
Identity_44IdentityRestoreV2:tensors:44"/device:CPU:0*
T0*
_output_shapes
:2
Identity_44?
AssignVariableOp_44AssignVariableOp)assignvariableop_44_nadam_output_kernel_mIdentity_44:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_44n
Identity_45IdentityRestoreV2:tensors:45"/device:CPU:0*
T0*
_output_shapes
:2
Identity_45?
AssignVariableOp_45AssignVariableOp'assignvariableop_45_nadam_output_bias_mIdentity_45:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_45n
Identity_46IdentityRestoreV2:tensors:46"/device:CPU:0*
T0*
_output_shapes
:2
Identity_46?
AssignVariableOp_46AssignVariableOp,assignvariableop_46_nadam_dense_105_kernel_vIdentity_46:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_46n
Identity_47IdentityRestoreV2:tensors:47"/device:CPU:0*
T0*
_output_shapes
:2
Identity_47?
AssignVariableOp_47AssignVariableOp*assignvariableop_47_nadam_dense_105_bias_vIdentity_47:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_47n
Identity_48IdentityRestoreV2:tensors:48"/device:CPU:0*
T0*
_output_shapes
:2
Identity_48?
AssignVariableOp_48AssignVariableOp,assignvariableop_48_nadam_dense_106_kernel_vIdentity_48:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_48n
Identity_49IdentityRestoreV2:tensors:49"/device:CPU:0*
T0*
_output_shapes
:2
Identity_49?
AssignVariableOp_49AssignVariableOp*assignvariableop_49_nadam_dense_106_bias_vIdentity_49:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_49n
Identity_50IdentityRestoreV2:tensors:50"/device:CPU:0*
T0*
_output_shapes
:2
Identity_50?
AssignVariableOp_50AssignVariableOp,assignvariableop_50_nadam_dense_107_kernel_vIdentity_50:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_50n
Identity_51IdentityRestoreV2:tensors:51"/device:CPU:0*
T0*
_output_shapes
:2
Identity_51?
AssignVariableOp_51AssignVariableOp*assignvariableop_51_nadam_dense_107_bias_vIdentity_51:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_51n
Identity_52IdentityRestoreV2:tensors:52"/device:CPU:0*
T0*
_output_shapes
:2
Identity_52?
AssignVariableOp_52AssignVariableOp,assignvariableop_52_nadam_dense_108_kernel_vIdentity_52:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_52n
Identity_53IdentityRestoreV2:tensors:53"/device:CPU:0*
T0*
_output_shapes
:2
Identity_53?
AssignVariableOp_53AssignVariableOp*assignvariableop_53_nadam_dense_108_bias_vIdentity_53:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_53n
Identity_54IdentityRestoreV2:tensors:54"/device:CPU:0*
T0*
_output_shapes
:2
Identity_54?
AssignVariableOp_54AssignVariableOp,assignvariableop_54_nadam_dense_109_kernel_vIdentity_54:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_54n
Identity_55IdentityRestoreV2:tensors:55"/device:CPU:0*
T0*
_output_shapes
:2
Identity_55?
AssignVariableOp_55AssignVariableOp*assignvariableop_55_nadam_dense_109_bias_vIdentity_55:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_55n
Identity_56IdentityRestoreV2:tensors:56"/device:CPU:0*
T0*
_output_shapes
:2
Identity_56?
AssignVariableOp_56AssignVariableOp,assignvariableop_56_nadam_dense_110_kernel_vIdentity_56:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_56n
Identity_57IdentityRestoreV2:tensors:57"/device:CPU:0*
T0*
_output_shapes
:2
Identity_57?
AssignVariableOp_57AssignVariableOp*assignvariableop_57_nadam_dense_110_bias_vIdentity_57:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_57n
Identity_58IdentityRestoreV2:tensors:58"/device:CPU:0*
T0*
_output_shapes
:2
Identity_58?
AssignVariableOp_58AssignVariableOp,assignvariableop_58_nadam_dense_111_kernel_vIdentity_58:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_58n
Identity_59IdentityRestoreV2:tensors:59"/device:CPU:0*
T0*
_output_shapes
:2
Identity_59?
AssignVariableOp_59AssignVariableOp*assignvariableop_59_nadam_dense_111_bias_vIdentity_59:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_59n
Identity_60IdentityRestoreV2:tensors:60"/device:CPU:0*
T0*
_output_shapes
:2
Identity_60?
AssignVariableOp_60AssignVariableOp)assignvariableop_60_nadam_output_kernel_vIdentity_60:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_60n
Identity_61IdentityRestoreV2:tensors:61"/device:CPU:0*
T0*
_output_shapes
:2
Identity_61?
AssignVariableOp_61AssignVariableOp'assignvariableop_61_nadam_output_bias_vIdentity_61:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_619
NoOpNoOp"/device:CPU:0*
_output_shapes
 2
NoOp?
Identity_62Identityfile_prefix^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_20^AssignVariableOp_21^AssignVariableOp_22^AssignVariableOp_23^AssignVariableOp_24^AssignVariableOp_25^AssignVariableOp_26^AssignVariableOp_27^AssignVariableOp_28^AssignVariableOp_29^AssignVariableOp_3^AssignVariableOp_30^AssignVariableOp_31^AssignVariableOp_32^AssignVariableOp_33^AssignVariableOp_34^AssignVariableOp_35^AssignVariableOp_36^AssignVariableOp_37^AssignVariableOp_38^AssignVariableOp_39^AssignVariableOp_4^AssignVariableOp_40^AssignVariableOp_41^AssignVariableOp_42^AssignVariableOp_43^AssignVariableOp_44^AssignVariableOp_45^AssignVariableOp_46^AssignVariableOp_47^AssignVariableOp_48^AssignVariableOp_49^AssignVariableOp_5^AssignVariableOp_50^AssignVariableOp_51^AssignVariableOp_52^AssignVariableOp_53^AssignVariableOp_54^AssignVariableOp_55^AssignVariableOp_56^AssignVariableOp_57^AssignVariableOp_58^AssignVariableOp_59^AssignVariableOp_6^AssignVariableOp_60^AssignVariableOp_61^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_9^NoOp"/device:CPU:0*
T0*
_output_shapes
: 2
Identity_62?
Identity_63IdentityIdentity_62:output:0^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_20^AssignVariableOp_21^AssignVariableOp_22^AssignVariableOp_23^AssignVariableOp_24^AssignVariableOp_25^AssignVariableOp_26^AssignVariableOp_27^AssignVariableOp_28^AssignVariableOp_29^AssignVariableOp_3^AssignVariableOp_30^AssignVariableOp_31^AssignVariableOp_32^AssignVariableOp_33^AssignVariableOp_34^AssignVariableOp_35^AssignVariableOp_36^AssignVariableOp_37^AssignVariableOp_38^AssignVariableOp_39^AssignVariableOp_4^AssignVariableOp_40^AssignVariableOp_41^AssignVariableOp_42^AssignVariableOp_43^AssignVariableOp_44^AssignVariableOp_45^AssignVariableOp_46^AssignVariableOp_47^AssignVariableOp_48^AssignVariableOp_49^AssignVariableOp_5^AssignVariableOp_50^AssignVariableOp_51^AssignVariableOp_52^AssignVariableOp_53^AssignVariableOp_54^AssignVariableOp_55^AssignVariableOp_56^AssignVariableOp_57^AssignVariableOp_58^AssignVariableOp_59^AssignVariableOp_6^AssignVariableOp_60^AssignVariableOp_61^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_9*
T0*
_output_shapes
: 2
Identity_63"#
identity_63Identity_63:output:0*?
_input_shapes?
?: ::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::2$
AssignVariableOpAssignVariableOp2(
AssignVariableOp_1AssignVariableOp_12*
AssignVariableOp_10AssignVariableOp_102*
AssignVariableOp_11AssignVariableOp_112*
AssignVariableOp_12AssignVariableOp_122*
AssignVariableOp_13AssignVariableOp_132*
AssignVariableOp_14AssignVariableOp_142*
AssignVariableOp_15AssignVariableOp_152*
AssignVariableOp_16AssignVariableOp_162*
AssignVariableOp_17AssignVariableOp_172*
AssignVariableOp_18AssignVariableOp_182*
AssignVariableOp_19AssignVariableOp_192(
AssignVariableOp_2AssignVariableOp_22*
AssignVariableOp_20AssignVariableOp_202*
AssignVariableOp_21AssignVariableOp_212*
AssignVariableOp_22AssignVariableOp_222*
AssignVariableOp_23AssignVariableOp_232*
AssignVariableOp_24AssignVariableOp_242*
AssignVariableOp_25AssignVariableOp_252*
AssignVariableOp_26AssignVariableOp_262*
AssignVariableOp_27AssignVariableOp_272*
AssignVariableOp_28AssignVariableOp_282*
AssignVariableOp_29AssignVariableOp_292(
AssignVariableOp_3AssignVariableOp_32*
AssignVariableOp_30AssignVariableOp_302*
AssignVariableOp_31AssignVariableOp_312*
AssignVariableOp_32AssignVariableOp_322*
AssignVariableOp_33AssignVariableOp_332*
AssignVariableOp_34AssignVariableOp_342*
AssignVariableOp_35AssignVariableOp_352*
AssignVariableOp_36AssignVariableOp_362*
AssignVariableOp_37AssignVariableOp_372*
AssignVariableOp_38AssignVariableOp_382*
AssignVariableOp_39AssignVariableOp_392(
AssignVariableOp_4AssignVariableOp_42*
AssignVariableOp_40AssignVariableOp_402*
AssignVariableOp_41AssignVariableOp_412*
AssignVariableOp_42AssignVariableOp_422*
AssignVariableOp_43AssignVariableOp_432*
AssignVariableOp_44AssignVariableOp_442*
AssignVariableOp_45AssignVariableOp_452*
AssignVariableOp_46AssignVariableOp_462*
AssignVariableOp_47AssignVariableOp_472*
AssignVariableOp_48AssignVariableOp_482*
AssignVariableOp_49AssignVariableOp_492(
AssignVariableOp_5AssignVariableOp_52*
AssignVariableOp_50AssignVariableOp_502*
AssignVariableOp_51AssignVariableOp_512*
AssignVariableOp_52AssignVariableOp_522*
AssignVariableOp_53AssignVariableOp_532*
AssignVariableOp_54AssignVariableOp_542*
AssignVariableOp_55AssignVariableOp_552*
AssignVariableOp_56AssignVariableOp_562*
AssignVariableOp_57AssignVariableOp_572*
AssignVariableOp_58AssignVariableOp_582*
AssignVariableOp_59AssignVariableOp_592(
AssignVariableOp_6AssignVariableOp_62*
AssignVariableOp_60AssignVariableOp_602*
AssignVariableOp_61AssignVariableOp_612(
AssignVariableOp_7AssignVariableOp_72(
AssignVariableOp_8AssignVariableOp_82(
AssignVariableOp_9AssignVariableOp_9:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix
?	
?
E__inference_dense_109_layer_call_and_return_conditional_losses_675510

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identity??BiasAdd/ReadVariableOp?MatMul/ReadVariableOp?
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource* 
_output_shapes
:
??*
dtype02
MatMul/ReadVariableOpt
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
MatMul?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2	
BiasAddY
ReluReluBiasAdd:output:0*
T0*(
_output_shapes
:??????????2
Relu?
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
T0*(
_output_shapes
:??????????2

Identity"
identityIdentity:output:0*/
_input_shapes
:??????????::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:P L
(
_output_shapes
:??????????
 
_user_specified_nameinputs
?	
?
E__inference_dense_110_layer_call_and_return_conditional_losses_675530

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identity??BiasAdd/ReadVariableOp?MatMul/ReadVariableOp?
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource* 
_output_shapes
:
??*
dtype02
MatMul/ReadVariableOpt
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
MatMul?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2	
BiasAddY
ReluReluBiasAdd:output:0*
T0*(
_output_shapes
:??????????2
Relu?
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
T0*(
_output_shapes
:??????????2

Identity"
identityIdentity:output:0*/
_input_shapes
:??????????::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:P L
(
_output_shapes
:??????????
 
_user_specified_nameinputs
?*
?
D__inference_model_15_layer_call_and_return_conditional_losses_674971
first_input
dense_105_674776
dense_105_674778
dense_106_674803
dense_106_674805
dense_107_674830
dense_107_674832
dense_108_674857
dense_108_674859
dense_109_674884
dense_109_674886
dense_110_674911
dense_110_674913
dense_111_674938
dense_111_674940
output_674965
output_674967
identity??!dense_105/StatefulPartitionedCall?!dense_106/StatefulPartitionedCall?!dense_107/StatefulPartitionedCall?!dense_108/StatefulPartitionedCall?!dense_109/StatefulPartitionedCall?!dense_110/StatefulPartitionedCall?!dense_111/StatefulPartitionedCall?output/StatefulPartitionedCall?
!dense_105/StatefulPartitionedCallStatefulPartitionedCallfirst_inputdense_105_674776dense_105_674778*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *N
fIRG
E__inference_dense_105_layer_call_and_return_conditional_losses_6747652#
!dense_105/StatefulPartitionedCall?
!dense_106/StatefulPartitionedCallStatefulPartitionedCall*dense_105/StatefulPartitionedCall:output:0dense_106_674803dense_106_674805*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *N
fIRG
E__inference_dense_106_layer_call_and_return_conditional_losses_6747922#
!dense_106/StatefulPartitionedCall?
!dense_107/StatefulPartitionedCallStatefulPartitionedCall*dense_106/StatefulPartitionedCall:output:0dense_107_674830dense_107_674832*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *N
fIRG
E__inference_dense_107_layer_call_and_return_conditional_losses_6748192#
!dense_107/StatefulPartitionedCall?
!dense_108/StatefulPartitionedCallStatefulPartitionedCall*dense_107/StatefulPartitionedCall:output:0dense_108_674857dense_108_674859*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *N
fIRG
E__inference_dense_108_layer_call_and_return_conditional_losses_6748462#
!dense_108/StatefulPartitionedCall?
!dense_109/StatefulPartitionedCallStatefulPartitionedCall*dense_108/StatefulPartitionedCall:output:0dense_109_674884dense_109_674886*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *N
fIRG
E__inference_dense_109_layer_call_and_return_conditional_losses_6748732#
!dense_109/StatefulPartitionedCall?
!dense_110/StatefulPartitionedCallStatefulPartitionedCall*dense_109/StatefulPartitionedCall:output:0dense_110_674911dense_110_674913*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *N
fIRG
E__inference_dense_110_layer_call_and_return_conditional_losses_6749002#
!dense_110/StatefulPartitionedCall?
!dense_111/StatefulPartitionedCallStatefulPartitionedCall*dense_110/StatefulPartitionedCall:output:0dense_111_674938dense_111_674940*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *N
fIRG
E__inference_dense_111_layer_call_and_return_conditional_losses_6749272#
!dense_111/StatefulPartitionedCall?
output/StatefulPartitionedCallStatefulPartitionedCall*dense_111/StatefulPartitionedCall:output:0output_674965output_674967*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????+*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *K
fFRD
B__inference_output_layer_call_and_return_conditional_losses_6749542 
output/StatefulPartitionedCall?
IdentityIdentity'output/StatefulPartitionedCall:output:0"^dense_105/StatefulPartitionedCall"^dense_106/StatefulPartitionedCall"^dense_107/StatefulPartitionedCall"^dense_108/StatefulPartitionedCall"^dense_109/StatefulPartitionedCall"^dense_110/StatefulPartitionedCall"^dense_111/StatefulPartitionedCall^output/StatefulPartitionedCall*
T0*'
_output_shapes
:?????????+2

Identity"
identityIdentity:output:0*f
_input_shapesU
S:?????????F::::::::::::::::2F
!dense_105/StatefulPartitionedCall!dense_105/StatefulPartitionedCall2F
!dense_106/StatefulPartitionedCall!dense_106/StatefulPartitionedCall2F
!dense_107/StatefulPartitionedCall!dense_107/StatefulPartitionedCall2F
!dense_108/StatefulPartitionedCall!dense_108/StatefulPartitionedCall2F
!dense_109/StatefulPartitionedCall!dense_109/StatefulPartitionedCall2F
!dense_110/StatefulPartitionedCall!dense_110/StatefulPartitionedCall2F
!dense_111/StatefulPartitionedCall!dense_111/StatefulPartitionedCall2@
output/StatefulPartitionedCalloutput/StatefulPartitionedCall:T P
'
_output_shapes
:?????????F
%
_user_specified_namefirst_input
?	
?
E__inference_dense_108_layer_call_and_return_conditional_losses_674846

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identity??BiasAdd/ReadVariableOp?MatMul/ReadVariableOp?
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource* 
_output_shapes
:
??*
dtype02
MatMul/ReadVariableOpt
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
MatMul?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2	
BiasAddY
ReluReluBiasAdd:output:0*
T0*(
_output_shapes
:??????????2
Relu?
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
T0*(
_output_shapes
:??????????2

Identity"
identityIdentity:output:0*/
_input_shapes
:??????????::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:P L
(
_output_shapes
:??????????
 
_user_specified_nameinputs
?*
?
D__inference_model_15_layer_call_and_return_conditional_losses_675062

inputs
dense_105_675021
dense_105_675023
dense_106_675026
dense_106_675028
dense_107_675031
dense_107_675033
dense_108_675036
dense_108_675038
dense_109_675041
dense_109_675043
dense_110_675046
dense_110_675048
dense_111_675051
dense_111_675053
output_675056
output_675058
identity??!dense_105/StatefulPartitionedCall?!dense_106/StatefulPartitionedCall?!dense_107/StatefulPartitionedCall?!dense_108/StatefulPartitionedCall?!dense_109/StatefulPartitionedCall?!dense_110/StatefulPartitionedCall?!dense_111/StatefulPartitionedCall?output/StatefulPartitionedCall?
!dense_105/StatefulPartitionedCallStatefulPartitionedCallinputsdense_105_675021dense_105_675023*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *N
fIRG
E__inference_dense_105_layer_call_and_return_conditional_losses_6747652#
!dense_105/StatefulPartitionedCall?
!dense_106/StatefulPartitionedCallStatefulPartitionedCall*dense_105/StatefulPartitionedCall:output:0dense_106_675026dense_106_675028*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *N
fIRG
E__inference_dense_106_layer_call_and_return_conditional_losses_6747922#
!dense_106/StatefulPartitionedCall?
!dense_107/StatefulPartitionedCallStatefulPartitionedCall*dense_106/StatefulPartitionedCall:output:0dense_107_675031dense_107_675033*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *N
fIRG
E__inference_dense_107_layer_call_and_return_conditional_losses_6748192#
!dense_107/StatefulPartitionedCall?
!dense_108/StatefulPartitionedCallStatefulPartitionedCall*dense_107/StatefulPartitionedCall:output:0dense_108_675036dense_108_675038*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *N
fIRG
E__inference_dense_108_layer_call_and_return_conditional_losses_6748462#
!dense_108/StatefulPartitionedCall?
!dense_109/StatefulPartitionedCallStatefulPartitionedCall*dense_108/StatefulPartitionedCall:output:0dense_109_675041dense_109_675043*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *N
fIRG
E__inference_dense_109_layer_call_and_return_conditional_losses_6748732#
!dense_109/StatefulPartitionedCall?
!dense_110/StatefulPartitionedCallStatefulPartitionedCall*dense_109/StatefulPartitionedCall:output:0dense_110_675046dense_110_675048*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *N
fIRG
E__inference_dense_110_layer_call_and_return_conditional_losses_6749002#
!dense_110/StatefulPartitionedCall?
!dense_111/StatefulPartitionedCallStatefulPartitionedCall*dense_110/StatefulPartitionedCall:output:0dense_111_675051dense_111_675053*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *N
fIRG
E__inference_dense_111_layer_call_and_return_conditional_losses_6749272#
!dense_111/StatefulPartitionedCall?
output/StatefulPartitionedCallStatefulPartitionedCall*dense_111/StatefulPartitionedCall:output:0output_675056output_675058*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????+*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *K
fFRD
B__inference_output_layer_call_and_return_conditional_losses_6749542 
output/StatefulPartitionedCall?
IdentityIdentity'output/StatefulPartitionedCall:output:0"^dense_105/StatefulPartitionedCall"^dense_106/StatefulPartitionedCall"^dense_107/StatefulPartitionedCall"^dense_108/StatefulPartitionedCall"^dense_109/StatefulPartitionedCall"^dense_110/StatefulPartitionedCall"^dense_111/StatefulPartitionedCall^output/StatefulPartitionedCall*
T0*'
_output_shapes
:?????????+2

Identity"
identityIdentity:output:0*f
_input_shapesU
S:?????????F::::::::::::::::2F
!dense_105/StatefulPartitionedCall!dense_105/StatefulPartitionedCall2F
!dense_106/StatefulPartitionedCall!dense_106/StatefulPartitionedCall2F
!dense_107/StatefulPartitionedCall!dense_107/StatefulPartitionedCall2F
!dense_108/StatefulPartitionedCall!dense_108/StatefulPartitionedCall2F
!dense_109/StatefulPartitionedCall!dense_109/StatefulPartitionedCall2F
!dense_110/StatefulPartitionedCall!dense_110/StatefulPartitionedCall2F
!dense_111/StatefulPartitionedCall!dense_111/StatefulPartitionedCall2@
output/StatefulPartitionedCalloutput/StatefulPartitionedCall:O K
'
_output_shapes
:?????????F
 
_user_specified_nameinputs
?

?
$__inference_signature_wrapper_675225
first_input
unknown
	unknown_0
	unknown_1
	unknown_2
	unknown_3
	unknown_4
	unknown_5
	unknown_6
	unknown_7
	unknown_8
	unknown_9

unknown_10

unknown_11

unknown_12

unknown_13

unknown_14
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallfirst_inputunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13
unknown_14*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????+*2
_read_only_resource_inputs
	
*-
config_proto

CPU

GPU 2J 8? **
f%R#
!__inference__wrapped_model_6747502
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:?????????+2

Identity"
identityIdentity:output:0*f
_input_shapesU
S:?????????F::::::::::::::::22
StatefulPartitionedCallStatefulPartitionedCall:T P
'
_output_shapes
:?????????F
%
_user_specified_namefirst_input
?

*__inference_dense_109_layer_call_fn_675519

inputs
unknown
	unknown_0
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *N
fIRG
E__inference_dense_109_layer_call_and_return_conditional_losses_6748732
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*(
_output_shapes
:??????????2

Identity"
identityIdentity:output:0*/
_input_shapes
:??????????::22
StatefulPartitionedCallStatefulPartitionedCall:P L
(
_output_shapes
:??????????
 
_user_specified_nameinputs
?

*__inference_dense_107_layer_call_fn_675479

inputs
unknown
	unknown_0
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *N
fIRG
E__inference_dense_107_layer_call_and_return_conditional_losses_6748192
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*(
_output_shapes
:??????????2

Identity"
identityIdentity:output:0*/
_input_shapes
:??????????::22
StatefulPartitionedCallStatefulPartitionedCall:P L
(
_output_shapes
:??????????
 
_user_specified_nameinputs
?L
?

D__inference_model_15_layer_call_and_return_conditional_losses_675285

inputs,
(dense_105_matmul_readvariableop_resource-
)dense_105_biasadd_readvariableop_resource,
(dense_106_matmul_readvariableop_resource-
)dense_106_biasadd_readvariableop_resource,
(dense_107_matmul_readvariableop_resource-
)dense_107_biasadd_readvariableop_resource,
(dense_108_matmul_readvariableop_resource-
)dense_108_biasadd_readvariableop_resource,
(dense_109_matmul_readvariableop_resource-
)dense_109_biasadd_readvariableop_resource,
(dense_110_matmul_readvariableop_resource-
)dense_110_biasadd_readvariableop_resource,
(dense_111_matmul_readvariableop_resource-
)dense_111_biasadd_readvariableop_resource)
%output_matmul_readvariableop_resource*
&output_biasadd_readvariableop_resource
identity?? dense_105/BiasAdd/ReadVariableOp?dense_105/MatMul/ReadVariableOp? dense_106/BiasAdd/ReadVariableOp?dense_106/MatMul/ReadVariableOp? dense_107/BiasAdd/ReadVariableOp?dense_107/MatMul/ReadVariableOp? dense_108/BiasAdd/ReadVariableOp?dense_108/MatMul/ReadVariableOp? dense_109/BiasAdd/ReadVariableOp?dense_109/MatMul/ReadVariableOp? dense_110/BiasAdd/ReadVariableOp?dense_110/MatMul/ReadVariableOp? dense_111/BiasAdd/ReadVariableOp?dense_111/MatMul/ReadVariableOp?output/BiasAdd/ReadVariableOp?output/MatMul/ReadVariableOp?
dense_105/MatMul/ReadVariableOpReadVariableOp(dense_105_matmul_readvariableop_resource*
_output_shapes
:	F?*
dtype02!
dense_105/MatMul/ReadVariableOp?
dense_105/MatMulMatMulinputs'dense_105/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
dense_105/MatMul?
 dense_105/BiasAdd/ReadVariableOpReadVariableOp)dense_105_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02"
 dense_105/BiasAdd/ReadVariableOp?
dense_105/BiasAddBiasAdddense_105/MatMul:product:0(dense_105/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
dense_105/BiasAddw
dense_105/ReluReludense_105/BiasAdd:output:0*
T0*(
_output_shapes
:??????????2
dense_105/Relu?
dense_106/MatMul/ReadVariableOpReadVariableOp(dense_106_matmul_readvariableop_resource* 
_output_shapes
:
??*
dtype02!
dense_106/MatMul/ReadVariableOp?
dense_106/MatMulMatMuldense_105/Relu:activations:0'dense_106/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
dense_106/MatMul?
 dense_106/BiasAdd/ReadVariableOpReadVariableOp)dense_106_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02"
 dense_106/BiasAdd/ReadVariableOp?
dense_106/BiasAddBiasAdddense_106/MatMul:product:0(dense_106/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
dense_106/BiasAddw
dense_106/ReluReludense_106/BiasAdd:output:0*
T0*(
_output_shapes
:??????????2
dense_106/Relu?
dense_107/MatMul/ReadVariableOpReadVariableOp(dense_107_matmul_readvariableop_resource* 
_output_shapes
:
??*
dtype02!
dense_107/MatMul/ReadVariableOp?
dense_107/MatMulMatMuldense_106/Relu:activations:0'dense_107/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
dense_107/MatMul?
 dense_107/BiasAdd/ReadVariableOpReadVariableOp)dense_107_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02"
 dense_107/BiasAdd/ReadVariableOp?
dense_107/BiasAddBiasAdddense_107/MatMul:product:0(dense_107/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
dense_107/BiasAddw
dense_107/ReluReludense_107/BiasAdd:output:0*
T0*(
_output_shapes
:??????????2
dense_107/Relu?
dense_108/MatMul/ReadVariableOpReadVariableOp(dense_108_matmul_readvariableop_resource* 
_output_shapes
:
??*
dtype02!
dense_108/MatMul/ReadVariableOp?
dense_108/MatMulMatMuldense_107/Relu:activations:0'dense_108/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
dense_108/MatMul?
 dense_108/BiasAdd/ReadVariableOpReadVariableOp)dense_108_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02"
 dense_108/BiasAdd/ReadVariableOp?
dense_108/BiasAddBiasAdddense_108/MatMul:product:0(dense_108/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
dense_108/BiasAddw
dense_108/ReluReludense_108/BiasAdd:output:0*
T0*(
_output_shapes
:??????????2
dense_108/Relu?
dense_109/MatMul/ReadVariableOpReadVariableOp(dense_109_matmul_readvariableop_resource* 
_output_shapes
:
??*
dtype02!
dense_109/MatMul/ReadVariableOp?
dense_109/MatMulMatMuldense_108/Relu:activations:0'dense_109/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
dense_109/MatMul?
 dense_109/BiasAdd/ReadVariableOpReadVariableOp)dense_109_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02"
 dense_109/BiasAdd/ReadVariableOp?
dense_109/BiasAddBiasAdddense_109/MatMul:product:0(dense_109/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
dense_109/BiasAddw
dense_109/ReluReludense_109/BiasAdd:output:0*
T0*(
_output_shapes
:??????????2
dense_109/Relu?
dense_110/MatMul/ReadVariableOpReadVariableOp(dense_110_matmul_readvariableop_resource* 
_output_shapes
:
??*
dtype02!
dense_110/MatMul/ReadVariableOp?
dense_110/MatMulMatMuldense_109/Relu:activations:0'dense_110/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
dense_110/MatMul?
 dense_110/BiasAdd/ReadVariableOpReadVariableOp)dense_110_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02"
 dense_110/BiasAdd/ReadVariableOp?
dense_110/BiasAddBiasAdddense_110/MatMul:product:0(dense_110/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
dense_110/BiasAddw
dense_110/ReluReludense_110/BiasAdd:output:0*
T0*(
_output_shapes
:??????????2
dense_110/Relu?
dense_111/MatMul/ReadVariableOpReadVariableOp(dense_111_matmul_readvariableop_resource* 
_output_shapes
:
??*
dtype02!
dense_111/MatMul/ReadVariableOp?
dense_111/MatMulMatMuldense_110/Relu:activations:0'dense_111/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
dense_111/MatMul?
 dense_111/BiasAdd/ReadVariableOpReadVariableOp)dense_111_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02"
 dense_111/BiasAdd/ReadVariableOp?
dense_111/BiasAddBiasAdddense_111/MatMul:product:0(dense_111/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
dense_111/BiasAddw
dense_111/ReluReludense_111/BiasAdd:output:0*
T0*(
_output_shapes
:??????????2
dense_111/Relu?
output/MatMul/ReadVariableOpReadVariableOp%output_matmul_readvariableop_resource*
_output_shapes
:	?+*
dtype02
output/MatMul/ReadVariableOp?
output/MatMulMatMuldense_111/Relu:activations:0$output/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????+2
output/MatMul?
output/BiasAdd/ReadVariableOpReadVariableOp&output_biasadd_readvariableop_resource*
_output_shapes
:+*
dtype02
output/BiasAdd/ReadVariableOp?
output/BiasAddBiasAddoutput/MatMul:product:0%output/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????+2
output/BiasAddv
output/SoftmaxSoftmaxoutput/BiasAdd:output:0*
T0*'
_output_shapes
:?????????+2
output/Softmax?
IdentityIdentityoutput/Softmax:softmax:0!^dense_105/BiasAdd/ReadVariableOp ^dense_105/MatMul/ReadVariableOp!^dense_106/BiasAdd/ReadVariableOp ^dense_106/MatMul/ReadVariableOp!^dense_107/BiasAdd/ReadVariableOp ^dense_107/MatMul/ReadVariableOp!^dense_108/BiasAdd/ReadVariableOp ^dense_108/MatMul/ReadVariableOp!^dense_109/BiasAdd/ReadVariableOp ^dense_109/MatMul/ReadVariableOp!^dense_110/BiasAdd/ReadVariableOp ^dense_110/MatMul/ReadVariableOp!^dense_111/BiasAdd/ReadVariableOp ^dense_111/MatMul/ReadVariableOp^output/BiasAdd/ReadVariableOp^output/MatMul/ReadVariableOp*
T0*'
_output_shapes
:?????????+2

Identity"
identityIdentity:output:0*f
_input_shapesU
S:?????????F::::::::::::::::2D
 dense_105/BiasAdd/ReadVariableOp dense_105/BiasAdd/ReadVariableOp2B
dense_105/MatMul/ReadVariableOpdense_105/MatMul/ReadVariableOp2D
 dense_106/BiasAdd/ReadVariableOp dense_106/BiasAdd/ReadVariableOp2B
dense_106/MatMul/ReadVariableOpdense_106/MatMul/ReadVariableOp2D
 dense_107/BiasAdd/ReadVariableOp dense_107/BiasAdd/ReadVariableOp2B
dense_107/MatMul/ReadVariableOpdense_107/MatMul/ReadVariableOp2D
 dense_108/BiasAdd/ReadVariableOp dense_108/BiasAdd/ReadVariableOp2B
dense_108/MatMul/ReadVariableOpdense_108/MatMul/ReadVariableOp2D
 dense_109/BiasAdd/ReadVariableOp dense_109/BiasAdd/ReadVariableOp2B
dense_109/MatMul/ReadVariableOpdense_109/MatMul/ReadVariableOp2D
 dense_110/BiasAdd/ReadVariableOp dense_110/BiasAdd/ReadVariableOp2B
dense_110/MatMul/ReadVariableOpdense_110/MatMul/ReadVariableOp2D
 dense_111/BiasAdd/ReadVariableOp dense_111/BiasAdd/ReadVariableOp2B
dense_111/MatMul/ReadVariableOpdense_111/MatMul/ReadVariableOp2>
output/BiasAdd/ReadVariableOpoutput/BiasAdd/ReadVariableOp2<
output/MatMul/ReadVariableOpoutput/MatMul/ReadVariableOp:O K
'
_output_shapes
:?????????F
 
_user_specified_nameinputs
?	
?
E__inference_dense_106_layer_call_and_return_conditional_losses_674792

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identity??BiasAdd/ReadVariableOp?MatMul/ReadVariableOp?
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource* 
_output_shapes
:
??*
dtype02
MatMul/ReadVariableOpt
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
MatMul?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2	
BiasAddY
ReluReluBiasAdd:output:0*
T0*(
_output_shapes
:??????????2
Relu?
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
T0*(
_output_shapes
:??????????2

Identity"
identityIdentity:output:0*/
_input_shapes
:??????????::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:P L
(
_output_shapes
:??????????
 
_user_specified_nameinputs
?
|
'__inference_output_layer_call_fn_675579

inputs
unknown
	unknown_0
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????+*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *K
fFRD
B__inference_output_layer_call_and_return_conditional_losses_6749542
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:?????????+2

Identity"
identityIdentity:output:0*/
_input_shapes
:??????????::22
StatefulPartitionedCallStatefulPartitionedCall:P L
(
_output_shapes
:??????????
 
_user_specified_nameinputs
?

*__inference_dense_110_layer_call_fn_675539

inputs
unknown
	unknown_0
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *N
fIRG
E__inference_dense_110_layer_call_and_return_conditional_losses_6749002
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*(
_output_shapes
:??????????2

Identity"
identityIdentity:output:0*/
_input_shapes
:??????????::22
StatefulPartitionedCallStatefulPartitionedCall:P L
(
_output_shapes
:??????????
 
_user_specified_nameinputs"?L
saver_filename:0StatefulPartitionedCall_1:0StatefulPartitionedCall_28"
saved_model_main_op

NoOp*>
__saved_model_init_op%#
__saved_model_init_op

NoOp*?
serving_default?
C
first_input4
serving_default_first_input:0?????????F:
output0
StatefulPartitionedCall:0?????????+tensorflow/serving/predict:??
?S
layer-0
layer_with_weights-0
layer-1
layer_with_weights-1
layer-2
layer_with_weights-2
layer-3
layer_with_weights-3
layer-4
layer_with_weights-4
layer-5
layer_with_weights-5
layer-6
layer_with_weights-6
layer-7
	layer_with_weights-7
	layer-8

	optimizer
regularization_losses
	variables
trainable_variables
	keras_api

signatures
+?&call_and_return_all_conditional_losses
?_default_save_signature
?__call__"?O
_tf_keras_network?N{"class_name": "Functional", "name": "model_15", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "must_restore_from_config": false, "config": {"name": "model_15", "layers": [{"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 70]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "first_input"}, "name": "first_input", "inbound_nodes": []}, {"class_name": "Dense", "config": {"name": "dense_105", "trainable": true, "dtype": "float32", "units": 128, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "dense_105", "inbound_nodes": [[["first_input", 0, 0, {}]]]}, {"class_name": "Dense", "config": {"name": "dense_106", "trainable": true, "dtype": "float32", "units": 256, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "dense_106", "inbound_nodes": [[["dense_105", 0, 0, {}]]]}, {"class_name": "Dense", "config": {"name": "dense_107", "trainable": true, "dtype": "float32", "units": 512, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "dense_107", "inbound_nodes": [[["dense_106", 0, 0, {}]]]}, {"class_name": "Dense", "config": {"name": "dense_108", "trainable": true, "dtype": "float32", "units": 512, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "dense_108", "inbound_nodes": [[["dense_107", 0, 0, {}]]]}, {"class_name": "Dense", "config": {"name": "dense_109", "trainable": true, "dtype": "float32", "units": 512, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "dense_109", "inbound_nodes": [[["dense_108", 0, 0, {}]]]}, {"class_name": "Dense", "config": {"name": "dense_110", "trainable": true, "dtype": "float32", "units": 256, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "dense_110", "inbound_nodes": [[["dense_109", 0, 0, {}]]]}, {"class_name": "Dense", "config": {"name": "dense_111", "trainable": true, "dtype": "float32", "units": 128, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "dense_111", "inbound_nodes": [[["dense_110", 0, 0, {}]]]}, {"class_name": "Dense", "config": {"name": "output", "trainable": true, "dtype": "float32", "units": 43, "activation": "softmax", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "output", "inbound_nodes": [[["dense_111", 0, 0, {}]]]}], "input_layers": [["first_input", 0, 0]], "output_layers": [["output", 0, 0]]}, "input_spec": [{"class_name": "InputSpec", "config": {"dtype": null, "shape": {"class_name": "__tuple__", "items": [null, 70]}, "ndim": 2, "max_ndim": null, "min_ndim": null, "axes": {}}}], "build_input_shape": {"class_name": "TensorShape", "items": [null, 70]}, "is_graph_network": true, "keras_version": "2.4.0", "backend": "tensorflow", "model_config": {"class_name": "Functional", "config": {"name": "model_15", "layers": [{"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 70]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "first_input"}, "name": "first_input", "inbound_nodes": []}, {"class_name": "Dense", "config": {"name": "dense_105", "trainable": true, "dtype": "float32", "units": 128, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "dense_105", "inbound_nodes": [[["first_input", 0, 0, {}]]]}, {"class_name": "Dense", "config": {"name": "dense_106", "trainable": true, "dtype": "float32", "units": 256, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "dense_106", "inbound_nodes": [[["dense_105", 0, 0, {}]]]}, {"class_name": "Dense", "config": {"name": "dense_107", "trainable": true, "dtype": "float32", "units": 512, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "dense_107", "inbound_nodes": [[["dense_106", 0, 0, {}]]]}, {"class_name": "Dense", "config": {"name": "dense_108", "trainable": true, "dtype": "float32", "units": 512, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "dense_108", "inbound_nodes": [[["dense_107", 0, 0, {}]]]}, {"class_name": "Dense", "config": {"name": "dense_109", "trainable": true, "dtype": "float32", "units": 512, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "dense_109", "inbound_nodes": [[["dense_108", 0, 0, {}]]]}, {"class_name": "Dense", "config": {"name": "dense_110", "trainable": true, "dtype": "float32", "units": 256, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "dense_110", "inbound_nodes": [[["dense_109", 0, 0, {}]]]}, {"class_name": "Dense", "config": {"name": "dense_111", "trainable": true, "dtype": "float32", "units": 128, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "dense_111", "inbound_nodes": [[["dense_110", 0, 0, {}]]]}, {"class_name": "Dense", "config": {"name": "output", "trainable": true, "dtype": "float32", "units": 43, "activation": "softmax", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "output", "inbound_nodes": [[["dense_111", 0, 0, {}]]]}], "input_layers": [["first_input", 0, 0]], "output_layers": [["output", 0, 0]]}}, "training_config": {"loss": "categorical_crossentropy", "metrics": [[{"class_name": "MeanMetricWrapper", "config": {"name": "accuracy", "dtype": "float32", "fn": "categorical_accuracy"}}, {"class_name": "TruePositives", "config": {"name": "true_positives_15", "dtype": "float32", "thresholds": null}}, {"class_name": "TrueNegatives", "config": {"name": "true_negatives_15", "dtype": "float32", "thresholds": null}}, {"class_name": "FalsePositives", "config": {"name": "false_positives_15", "dtype": "float32", "thresholds": null}}, {"class_name": "FalseNegatives", "config": {"name": "false_negatives_15", "dtype": "float32", "thresholds": null}}]], "weighted_metrics": null, "loss_weights": null, "optimizer_config": {"class_name": "Nadam", "config": {"name": "Nadam", "learning_rate": 0.0010000000474974513, "decay": 0.004000000189989805, "beta_1": 0.8999999761581421, "beta_2": 0.9990000128746033, "epsilon": 1e-07}}}}
?"?
_tf_keras_input_layer?{"class_name": "InputLayer", "name": "first_input", "dtype": "float32", "sparse": false, "ragged": false, "batch_input_shape": {"class_name": "__tuple__", "items": [null, 70]}, "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 70]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "first_input"}}
?

kernel
bias
regularization_losses
	variables
trainable_variables
	keras_api
+?&call_and_return_all_conditional_losses
?__call__"?
_tf_keras_layer?{"class_name": "Dense", "name": "dense_105", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "dense_105", "trainable": true, "dtype": "float32", "units": 128, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 70}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 70]}}
?

kernel
bias
regularization_losses
	variables
trainable_variables
	keras_api
+?&call_and_return_all_conditional_losses
?__call__"?
_tf_keras_layer?{"class_name": "Dense", "name": "dense_106", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "dense_106", "trainable": true, "dtype": "float32", "units": 256, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 128}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 128]}}
?

kernel
bias
regularization_losses
	variables
 trainable_variables
!	keras_api
+?&call_and_return_all_conditional_losses
?__call__"?
_tf_keras_layer?{"class_name": "Dense", "name": "dense_107", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "dense_107", "trainable": true, "dtype": "float32", "units": 512, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 256}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 256]}}
?

"kernel
#bias
$regularization_losses
%	variables
&trainable_variables
'	keras_api
+?&call_and_return_all_conditional_losses
?__call__"?
_tf_keras_layer?{"class_name": "Dense", "name": "dense_108", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "dense_108", "trainable": true, "dtype": "float32", "units": 512, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 512}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 512]}}
?

(kernel
)bias
*regularization_losses
+	variables
,trainable_variables
-	keras_api
+?&call_and_return_all_conditional_losses
?__call__"?
_tf_keras_layer?{"class_name": "Dense", "name": "dense_109", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "dense_109", "trainable": true, "dtype": "float32", "units": 512, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 512}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 512]}}
?

.kernel
/bias
0regularization_losses
1	variables
2trainable_variables
3	keras_api
+?&call_and_return_all_conditional_losses
?__call__"?
_tf_keras_layer?{"class_name": "Dense", "name": "dense_110", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "dense_110", "trainable": true, "dtype": "float32", "units": 256, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 512}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 512]}}
?

4kernel
5bias
6regularization_losses
7	variables
8trainable_variables
9	keras_api
+?&call_and_return_all_conditional_losses
?__call__"?
_tf_keras_layer?{"class_name": "Dense", "name": "dense_111", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "dense_111", "trainable": true, "dtype": "float32", "units": 128, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 256}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 256]}}
?

:kernel
;bias
<regularization_losses
=	variables
>trainable_variables
?	keras_api
+?&call_and_return_all_conditional_losses
?__call__"?
_tf_keras_layer?{"class_name": "Dense", "name": "output", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "output", "trainable": true, "dtype": "float32", "units": 43, "activation": "softmax", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 128}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 128]}}
?
@iter

Abeta_1

Bbeta_2
	Cdecay
Dlearning_rate
Emomentum_cachem?m?m?m?m?m?"m?#m?(m?)m?.m?/m?4m?5m?:m?;m?v?v?v?v?v?v?"v?#v?(v?)v?.v?/v?4v?5v?:v?;v?"
	optimizer
 "
trackable_list_wrapper
?
0
1
2
3
4
5
"6
#7
(8
)9
.10
/11
412
513
:14
;15"
trackable_list_wrapper
?
0
1
2
3
4
5
"6
#7
(8
)9
.10
/11
412
513
:14
;15"
trackable_list_wrapper
?

Flayers
Gmetrics
Hnon_trainable_variables
regularization_losses
Ilayer_regularization_losses
Jlayer_metrics
	variables
trainable_variables
?__call__
?_default_save_signature
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
-
?serving_default"
signature_map
#:!	F?2dense_105/kernel
:?2dense_105/bias
 "
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
?

Klayers
Lnon_trainable_variables
regularization_losses
	variables
Mlayer_regularization_losses
Nlayer_metrics
Ometrics
trainable_variables
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
$:"
??2dense_106/kernel
:?2dense_106/bias
 "
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
?

Players
Qnon_trainable_variables
regularization_losses
	variables
Rlayer_regularization_losses
Slayer_metrics
Tmetrics
trainable_variables
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
$:"
??2dense_107/kernel
:?2dense_107/bias
 "
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
?

Ulayers
Vnon_trainable_variables
regularization_losses
	variables
Wlayer_regularization_losses
Xlayer_metrics
Ymetrics
 trainable_variables
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
$:"
??2dense_108/kernel
:?2dense_108/bias
 "
trackable_list_wrapper
.
"0
#1"
trackable_list_wrapper
.
"0
#1"
trackable_list_wrapper
?

Zlayers
[non_trainable_variables
$regularization_losses
%	variables
\layer_regularization_losses
]layer_metrics
^metrics
&trainable_variables
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
$:"
??2dense_109/kernel
:?2dense_109/bias
 "
trackable_list_wrapper
.
(0
)1"
trackable_list_wrapper
.
(0
)1"
trackable_list_wrapper
?

_layers
`non_trainable_variables
*regularization_losses
+	variables
alayer_regularization_losses
blayer_metrics
cmetrics
,trainable_variables
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
$:"
??2dense_110/kernel
:?2dense_110/bias
 "
trackable_list_wrapper
.
.0
/1"
trackable_list_wrapper
.
.0
/1"
trackable_list_wrapper
?

dlayers
enon_trainable_variables
0regularization_losses
1	variables
flayer_regularization_losses
glayer_metrics
hmetrics
2trainable_variables
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
$:"
??2dense_111/kernel
:?2dense_111/bias
 "
trackable_list_wrapper
.
40
51"
trackable_list_wrapper
.
40
51"
trackable_list_wrapper
?

ilayers
jnon_trainable_variables
6regularization_losses
7	variables
klayer_regularization_losses
llayer_metrics
mmetrics
8trainable_variables
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
 :	?+2output/kernel
:+2output/bias
 "
trackable_list_wrapper
.
:0
;1"
trackable_list_wrapper
.
:0
;1"
trackable_list_wrapper
?

nlayers
onon_trainable_variables
<regularization_losses
=	variables
player_regularization_losses
qlayer_metrics
rmetrics
>trainable_variables
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
:	 (2
Nadam/iter
: (2Nadam/beta_1
: (2Nadam/beta_2
: (2Nadam/decay
: (2Nadam/learning_rate
: (2Nadam/momentum_cache
_
0
1
2
3
4
5
6
7
	8"
trackable_list_wrapper
J
s0
t1
u2
v3
w4
x5"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
?
	ytotal
	zcount
{	variables
|	keras_api"?
_tf_keras_metricj{"class_name": "Mean", "name": "loss", "dtype": "float32", "config": {"name": "loss", "dtype": "float32"}}
?
	}total
	~count

_fn_kwargs
?	variables
?	keras_api"?
_tf_keras_metric?{"class_name": "MeanMetricWrapper", "name": "accuracy", "dtype": "float32", "config": {"name": "accuracy", "dtype": "float32", "fn": "categorical_accuracy"}}
?
?
thresholds
?accumulator
?	variables
?	keras_api"?
_tf_keras_metric?{"class_name": "TruePositives", "name": "true_positives_15", "dtype": "float32", "config": {"name": "true_positives_15", "dtype": "float32", "thresholds": null}}
?
?
thresholds
?accumulator
?	variables
?	keras_api"?
_tf_keras_metric?{"class_name": "TrueNegatives", "name": "true_negatives_15", "dtype": "float32", "config": {"name": "true_negatives_15", "dtype": "float32", "thresholds": null}}
?
?
thresholds
?accumulator
?	variables
?	keras_api"?
_tf_keras_metric?{"class_name": "FalsePositives", "name": "false_positives_15", "dtype": "float32", "config": {"name": "false_positives_15", "dtype": "float32", "thresholds": null}}
?
?
thresholds
?accumulator
?	variables
?	keras_api"?
_tf_keras_metric?{"class_name": "FalseNegatives", "name": "false_negatives_15", "dtype": "float32", "config": {"name": "false_negatives_15", "dtype": "float32", "thresholds": null}}
:  (2total
:  (2count
.
y0
z1"
trackable_list_wrapper
-
{	variables"
_generic_user_object
:  (2total
:  (2count
 "
trackable_dict_wrapper
.
}0
~1"
trackable_list_wrapper
.
?	variables"
_generic_user_object
 "
trackable_list_wrapper
: (2accumulator
(
?0"
trackable_list_wrapper
.
?	variables"
_generic_user_object
 "
trackable_list_wrapper
: (2accumulator
(
?0"
trackable_list_wrapper
.
?	variables"
_generic_user_object
 "
trackable_list_wrapper
: (2accumulator
(
?0"
trackable_list_wrapper
.
?	variables"
_generic_user_object
 "
trackable_list_wrapper
: (2accumulator
(
?0"
trackable_list_wrapper
.
?	variables"
_generic_user_object
):'	F?2Nadam/dense_105/kernel/m
#:!?2Nadam/dense_105/bias/m
*:(
??2Nadam/dense_106/kernel/m
#:!?2Nadam/dense_106/bias/m
*:(
??2Nadam/dense_107/kernel/m
#:!?2Nadam/dense_107/bias/m
*:(
??2Nadam/dense_108/kernel/m
#:!?2Nadam/dense_108/bias/m
*:(
??2Nadam/dense_109/kernel/m
#:!?2Nadam/dense_109/bias/m
*:(
??2Nadam/dense_110/kernel/m
#:!?2Nadam/dense_110/bias/m
*:(
??2Nadam/dense_111/kernel/m
#:!?2Nadam/dense_111/bias/m
&:$	?+2Nadam/output/kernel/m
:+2Nadam/output/bias/m
):'	F?2Nadam/dense_105/kernel/v
#:!?2Nadam/dense_105/bias/v
*:(
??2Nadam/dense_106/kernel/v
#:!?2Nadam/dense_106/bias/v
*:(
??2Nadam/dense_107/kernel/v
#:!?2Nadam/dense_107/bias/v
*:(
??2Nadam/dense_108/kernel/v
#:!?2Nadam/dense_108/bias/v
*:(
??2Nadam/dense_109/kernel/v
#:!?2Nadam/dense_109/bias/v
*:(
??2Nadam/dense_110/kernel/v
#:!?2Nadam/dense_110/bias/v
*:(
??2Nadam/dense_111/kernel/v
#:!?2Nadam/dense_111/bias/v
&:$	?+2Nadam/output/kernel/v
:+2Nadam/output/bias/v
?2?
D__inference_model_15_layer_call_and_return_conditional_losses_675345
D__inference_model_15_layer_call_and_return_conditional_losses_674971
D__inference_model_15_layer_call_and_return_conditional_losses_675285
D__inference_model_15_layer_call_and_return_conditional_losses_675015?
???
FullArgSpec1
args)?&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults?
p 

 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?2?
!__inference__wrapped_model_674750?
???
FullArgSpec
args? 
varargsjargs
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? **?'
%?"
first_input?????????F
?2?
)__inference_model_15_layer_call_fn_675419
)__inference_model_15_layer_call_fn_675382
)__inference_model_15_layer_call_fn_675178
)__inference_model_15_layer_call_fn_675097?
???
FullArgSpec1
args)?&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults?
p 

 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?2?
E__inference_dense_105_layer_call_and_return_conditional_losses_675430?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
*__inference_dense_105_layer_call_fn_675439?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
E__inference_dense_106_layer_call_and_return_conditional_losses_675450?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
*__inference_dense_106_layer_call_fn_675459?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
E__inference_dense_107_layer_call_and_return_conditional_losses_675470?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
*__inference_dense_107_layer_call_fn_675479?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
E__inference_dense_108_layer_call_and_return_conditional_losses_675490?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
*__inference_dense_108_layer_call_fn_675499?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
E__inference_dense_109_layer_call_and_return_conditional_losses_675510?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
*__inference_dense_109_layer_call_fn_675519?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
E__inference_dense_110_layer_call_and_return_conditional_losses_675530?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
*__inference_dense_110_layer_call_fn_675539?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
E__inference_dense_111_layer_call_and_return_conditional_losses_675550?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
*__inference_dense_111_layer_call_fn_675559?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
B__inference_output_layer_call_and_return_conditional_losses_675570?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
'__inference_output_layer_call_fn_675579?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?B?
$__inference_signature_wrapper_675225first_input"?
???
FullArgSpec
args? 
varargs
 
varkwjkwargs
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 ?
!__inference__wrapped_model_674750y"#()./45:;4?1
*?'
%?"
first_input?????????F
? "/?,
*
output ?
output?????????+?
E__inference_dense_105_layer_call_and_return_conditional_losses_675430]/?,
%?"
 ?
inputs?????????F
? "&?#
?
0??????????
? ~
*__inference_dense_105_layer_call_fn_675439P/?,
%?"
 ?
inputs?????????F
? "????????????
E__inference_dense_106_layer_call_and_return_conditional_losses_675450^0?-
&?#
!?
inputs??????????
? "&?#
?
0??????????
? 
*__inference_dense_106_layer_call_fn_675459Q0?-
&?#
!?
inputs??????????
? "????????????
E__inference_dense_107_layer_call_and_return_conditional_losses_675470^0?-
&?#
!?
inputs??????????
? "&?#
?
0??????????
? 
*__inference_dense_107_layer_call_fn_675479Q0?-
&?#
!?
inputs??????????
? "????????????
E__inference_dense_108_layer_call_and_return_conditional_losses_675490^"#0?-
&?#
!?
inputs??????????
? "&?#
?
0??????????
? 
*__inference_dense_108_layer_call_fn_675499Q"#0?-
&?#
!?
inputs??????????
? "????????????
E__inference_dense_109_layer_call_and_return_conditional_losses_675510^()0?-
&?#
!?
inputs??????????
? "&?#
?
0??????????
? 
*__inference_dense_109_layer_call_fn_675519Q()0?-
&?#
!?
inputs??????????
? "????????????
E__inference_dense_110_layer_call_and_return_conditional_losses_675530^./0?-
&?#
!?
inputs??????????
? "&?#
?
0??????????
? 
*__inference_dense_110_layer_call_fn_675539Q./0?-
&?#
!?
inputs??????????
? "????????????
E__inference_dense_111_layer_call_and_return_conditional_losses_675550^450?-
&?#
!?
inputs??????????
? "&?#
?
0??????????
? 
*__inference_dense_111_layer_call_fn_675559Q450?-
&?#
!?
inputs??????????
? "????????????
D__inference_model_15_layer_call_and_return_conditional_losses_674971w"#()./45:;<?9
2?/
%?"
first_input?????????F
p

 
? "%?"
?
0?????????+
? ?
D__inference_model_15_layer_call_and_return_conditional_losses_675015w"#()./45:;<?9
2?/
%?"
first_input?????????F
p 

 
? "%?"
?
0?????????+
? ?
D__inference_model_15_layer_call_and_return_conditional_losses_675285r"#()./45:;7?4
-?*
 ?
inputs?????????F
p

 
? "%?"
?
0?????????+
? ?
D__inference_model_15_layer_call_and_return_conditional_losses_675345r"#()./45:;7?4
-?*
 ?
inputs?????????F
p 

 
? "%?"
?
0?????????+
? ?
)__inference_model_15_layer_call_fn_675097j"#()./45:;<?9
2?/
%?"
first_input?????????F
p

 
? "??????????+?
)__inference_model_15_layer_call_fn_675178j"#()./45:;<?9
2?/
%?"
first_input?????????F
p 

 
? "??????????+?
)__inference_model_15_layer_call_fn_675382e"#()./45:;7?4
-?*
 ?
inputs?????????F
p

 
? "??????????+?
)__inference_model_15_layer_call_fn_675419e"#()./45:;7?4
-?*
 ?
inputs?????????F
p 

 
? "??????????+?
B__inference_output_layer_call_and_return_conditional_losses_675570]:;0?-
&?#
!?
inputs??????????
? "%?"
?
0?????????+
? {
'__inference_output_layer_call_fn_675579P:;0?-
&?#
!?
inputs??????????
? "??????????+?
$__inference_signature_wrapper_675225?"#()./45:;C?@
? 
9?6
4
first_input%?"
first_input?????????F"/?,
*
output ?
output?????????+