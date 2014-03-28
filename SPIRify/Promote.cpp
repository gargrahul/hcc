//===- Hello.cpp - Example code from "Writing an LLVM Pass" ---------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file implements two versions of the LLVM "Hello World" pass described
// in docs/WritingAnLLVMPass.html
//
//===----------------------------------------------------------------------===//

#define DEBUG_TYPE "PromoteGlobals"

#include "Promote.h"

#include "llvm/ADT/Twine.h"
#include "llvm/Analysis/CallGraph.h"
#include "llvm/Analysis/Verifier.h"
#include "llvm/IR/DataLayout.h"
#include "llvm/IR/IntrinsicInst.h"
#include "llvm/IR/Metadata.h"
#include "llvm/IR/Module.h"
#include "llvm/Pass.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/Transforms/Utils/Cloning.h"
#include <map>
#include <set>
using namespace llvm;
#ifdef __APPLE__
#define TILE_STATIC_NAME "clamp,opencl_local"
#else
#define TILE_STATIC_NAME "clamp_opencl_local"
#endif

namespace {

/* Data type container into which the list of LLVM functions
   that are OpenCL kernels will be stored */
typedef SmallVector<Function *, 3> FunctionVect;
typedef SmallVector<std::pair<Function *, Function *>, 3> FunctionPairVect;
typedef std::map <Function *, Function *> FunctionMap;

/* The name of the MDNode into which the list of
   MD nodes referencing each OpenCL kernel is stored. */
static Twine KernelListMDNodeName = "opencl.kernels";

enum {
        GlobalAddressSpace = 1,
        LocalAddressSpace = 3
};

class InstUpdateWorkList;

class InstUpdate {
        public:
        virtual ~InstUpdate() {}
        virtual void operator()(InstUpdateWorkList*) =0;
};

void updateInstructionWithNewOperand(Instruction * I, 
                                     Value * oldOperand, 
                                     Value * newOperand,
                                     InstUpdateWorkList * updatesNeeded);

void updateListWithUsers ( Value::use_iterator U, const Value::use_iterator& Ue, 
                           Value * oldOperand, Value * newOperand,
                           InstUpdateWorkList * updates );
/* This structure hold information on which instruction
   should be updated with a new operand. The subject is
   the instruction that need to be updated. OldOperand
   is the current operand within the subject that needs
   to be updated. The newOperand is the value that needs
   to be substituted for the oldOperand value. The Old
   operand value is provided to identify which operand
   needs to be updated, as it may not be trivial to
   identify which operand is affected. In addition, the same
   value may be used multiple times by an instruction.
   Rather than allocating this structure multiple time,
   only one is used. */
class ForwardUpdate : public InstUpdate {
        public:
        ForwardUpdate(Instruction * s, Value * oldOperand, Value * newOperand);
        void operator()(InstUpdateWorkList *);
        static void UpdateUsersWithNewOperand(Instruction *s, Value * oldOperand, Value * newOperand, InstUpdateWorkList * workList);
        private:
        Instruction * subject;
        Value * oldOperand;
        Value * newOperand;
};

ForwardUpdate::ForwardUpdate(Instruction *s, 
                             Value *oldOp, Value *newOp)
                             : subject(s), oldOperand(oldOp), newOperand(newOp)
{}

void ForwardUpdate::operator()(InstUpdateWorkList * workList)
{
        DEBUG(llvm::errs() << "F: "; subject->dump(););
        updateInstructionWithNewOperand (subject,
                                         oldOperand,
                                         newOperand,
                                         workList); 
       // Instruction *oldInst = dyn_cast<Instruction>(oldOperand);
       // if ( oldInst->use_empty () ) {
                //oldInst->eraseFromParent ();
       // }
}

void ForwardUpdate::UpdateUsersWithNewOperand (Instruction * Insn, 
                                               Value * oldOperand,
                                               Value * newOperand,
                                               InstUpdateWorkList * workList)
{
        updateListWithUsers ( Insn->use_begin (), Insn->use_end(),
                              oldOperand, newOperand,
                              workList );
} 

class BackwardUpdate : public InstUpdate {
        public:
        BackwardUpdate(Instruction * upstream, Type* expected);
        void operator()(InstUpdateWorkList *updater);
        static void setExpectedType(Instruction * Insn, Type * expected, 
                                    InstUpdateWorkList * updater);
        private:
        Instruction * upstream;
        Type * expected;
};

class InstUpdateWorkList {
        public:
        ~InstUpdateWorkList();
        bool run();
        bool empty() const;
        void addUpdate(InstUpdate * update);
        private:
        typedef std::vector<InstUpdate *> WorkListTy;
        WorkListTy workList;
};

BackwardUpdate::BackwardUpdate (Instruction * insn, Type *type)
                                : upstream (insn), expected (type)
{}

void updateBackAllocaInst(AllocaInst * AI, Type * expected, 
                          InstUpdateWorkList * updater)
{
        PointerType * ptrType = dyn_cast<PointerType> (expected);
        if ( !ptrType ) {
                DEBUG(llvm::errs() << "Was expecting a pointer type. Got ";
                      expected->dump(););
        }        

        AllocaInst * newInst = new AllocaInst (ptrType->getElementType(),
                                               AI->getArraySize (), "",
                                               AI);

        ForwardUpdate::UpdateUsersWithNewOperand (AI, AI, newInst,
                                                  updater); 
}

Type * patchType ( Type * baseType, Type* patch, User::op_iterator idx, User::op_iterator idx_end);

Type * patchType ( Type * baseType, Type* patch, User::op_iterator idx, User::op_iterator idx_end)
{
        if ( idx == idx_end ) {
                return patch;
        }

        bool isIndexLiteral = false;
        uint64_t literalIndex = 0;
        if ( ConstantInt * CI = dyn_cast<ConstantInt>(*idx) ) {
                literalIndex = CI->getZExtValue ();
                isIndexLiteral = true;
        }

        if ( StructType * ST = dyn_cast<StructType>(baseType ) ) {
                if ( !isIndexLiteral ) {
                        llvm::errs() << "Expecting literal index for struct type\n";
                        return NULL;
                }
                std::vector<Type *> newElements;
                for (unsigned elem = 0, last_elem = ST->getNumElements();
                     elem != last_elem; ++elem) {
                        Type * elementType = ST->getElementType (elem);
                        if ( elem != literalIndex ) {
                                newElements.push_back (elementType);
                                continue;
                        }
                        Type * transformed = patchType (elementType, patch,
                                                        ++idx, idx_end);
                        newElements.push_back (transformed);
                }
                return StructType::get (ST->getContext(),
                                        ArrayRef<Type *>(newElements),
                                        ST->isPacked());
       }
       DEBUG(llvm::errs() << "Patch type not handling ";
             baseType->dump(););
       return NULL;
}

void updateBackGEP (GetElementPtrInst * GEP, Type* expected,
                    InstUpdateWorkList * updater) 
{
        PointerType * ptrExpected = dyn_cast<PointerType> (expected);
        if ( !ptrExpected ) {
                llvm::errs() << "Expected type for GEP is not a pointer!\n";
                return;
        }
        PointerType * ptrSource = 
                dyn_cast<PointerType> (GEP->getPointerOperand()->getType());
        if ( !ptrSource ) {
                llvm::errs() << "Source operand type is not a pointer!\n";
                return;
        }
        User::op_iterator first_idx = GEP->idx_begin();
        ++first_idx;
        Type * newElementType = patchType (ptrSource->getElementType(),
                                           ptrExpected->getElementType(), 
                                           first_idx, GEP->idx_end());

        PointerType  * newUpstreamType = 
                PointerType::get(newElementType,
                                 ptrExpected->getAddressSpace());
        Instruction * ptrProducer = 
                dyn_cast<Instruction>(GEP->getPointerOperand());
        assert(ptrProducer 
               && "Was expecting an instruction as source operand for GEP");
        BackwardUpdate::setExpectedType (ptrProducer,
                                         newUpstreamType, updater);
}

void updateBackLoad (LoadInst * L, Type * expected, 
                     InstUpdateWorkList * updater)
{
        Value * ptrOperand = L->getPointerOperand();
        Instruction * ptrSource = dyn_cast<Instruction>(ptrOperand);

        assert(ptrSource
               && "Was expecting an instruction for the source operand");

        PointerType * sourceType = 
                dyn_cast<PointerType> (ptrOperand->getType());
        assert(sourceType
               && "Load ptr operand's type is not a pointer type");

        PointerType * newPtrType = 
                PointerType::get(expected, 
                                 sourceType->getAddressSpace());

        BackwardUpdate::setExpectedType(ptrSource, newPtrType, updater);
}

void updateBackBitCast (BitCastInst * BCI, Type * expected, 
                        InstUpdateWorkList * updater)
{
        DEBUG(BCI->dump(););
        Type * srcType = BCI->getSrcTy();
        PointerType * ptrSrcType = dyn_cast<PointerType> (srcType);
        assert (ptrSrcType 
                && "Unexpected non-pointer type as source operand of bitcast");

        Type * destType = BCI->getDestTy();
        PointerType * ptrDestType = dyn_cast<PointerType> (destType);
        assert (ptrDestType
                && "Unexpected non-pointer type as dest operand of bitcast");

        Type * srcElement = ptrSrcType->getElementType ();
        StructType *srcElementStructType = dyn_cast<StructType> (srcElement);
        if ( !srcElementStructType ) {
                 DEBUG(llvm::errs () << "Do not know how handle bitcast\n";);
                 return;
        }

        Type * dstElement = ptrDestType->getElementType ();
        StructType *dstElementStructType = dyn_cast<StructType> (dstElement);
        if ( !dstElementStructType ) {
                 DEBUG(llvm::errs () << "Do not know how handle bitcast\n";);
                 return;
        }

        bool sameLayout = 
                srcElementStructType->isLayoutIdentical(dstElementStructType);
        if ( !sameLayout ) {
                DEBUG(llvm::errs() << "Different layout in bitcast!\n";);
                return;
        }

        Instruction *sourceOperand = dyn_cast<Instruction>(BCI->getOperand(0));
        if ( !sourceOperand ) {
                DEBUG(llvm::errs() << "Do not know how to handle"
                                      " non-instruction source operand\n";);
        }
        BitCastInst * newBitCast = 
                new BitCastInst(sourceOperand,
                                expected, "", BCI);

        ForwardUpdate::UpdateUsersWithNewOperand (BCI, BCI, newBitCast,
                                                  updater); 
        BackwardUpdate::setExpectedType(sourceOperand, expected, updater);
        return;


}

void BackwardUpdate::operator ()(InstUpdateWorkList *updater)
{
        DEBUG(llvm::errs() << "B: "; upstream->dump(););
        if ( AllocaInst * AI = dyn_cast<AllocaInst> (upstream) ) {
                updateBackAllocaInst (AI, expected, updater);
                return;
        }
        if ( GetElementPtrInst * GEP = dyn_cast<GetElementPtrInst>(upstream) ) {
                updateBackGEP (GEP, expected, updater);
                 return;
        }
        if ( LoadInst * LI = dyn_cast<LoadInst>(upstream) ) {
                updateBackLoad (LI, expected, updater);
                return;
        } 
        if ( BitCastInst * BCI = dyn_cast<BitCastInst>(upstream) ) {
                updateBackBitCast (BCI, expected, updater);
                return;
        }
        DEBUG(llvm::errs() << "Do not know how to update ";
              upstream->dump(); llvm::errs() << " with "; expected->dump();
              llvm::errs() << "\n";);
        return;

}

void BackwardUpdate::setExpectedType (Instruction * Insn, Type * expected,
                                      InstUpdateWorkList * update)
{
        update->addUpdate (new BackwardUpdate (Insn, expected));
}



bool InstUpdateWorkList::run()
{
        bool didSomething = false;
        while( !workList.empty() ) {
                InstUpdate * update = workList.back();
                workList.pop_back();
                (*update) (this);
                delete update;
                didSomething = true;
        }
        return didSomething;
}

bool InstUpdateWorkList::empty() const
{
        return workList.empty ();
}

void InstUpdateWorkList::addUpdate(InstUpdate * update)
{
        workList.push_back(update);
}

InstUpdateWorkList::~InstUpdateWorkList ()
{
        for ( WorkListTy::iterator U = workList.begin(), Ue = workList.end();
              U != Ue; ++U ) {
                delete *U;
        }
}
Function * createPromotedFunctionToType ( Function * F, FunctionType * promoteType);

/* Find the MDNode which reference the list of opencl kernels.
   NULL if it does not exists. */

NamedMDNode * getKernelListMDNode (Module & M)
{
        return M.getNamedMetadata(KernelListMDNodeName);
}

NamedMDNode * getNewKernelListMDNode (Module & M)
{
        NamedMDNode * current = getKernelListMDNode (M);
        if ( current ) {
                M.eraseNamedMetadata (current);
        }
        return M.getOrInsertNamedMetadata (KernelListMDNodeName.str());
}

Type * mapTypeToGlobal ( Type *);

StructType * mapTypeToGlobal ( StructType * T) {
        std::vector<Type *> translatedTypes;

        for (unsigned elem = 0, last_elem = T->getNumElements();
             elem != last_elem; ++elem) {
                Type * baseType = T->getElementType (elem);
                Type * translatedType = mapTypeToGlobal (baseType);
                translatedTypes.push_back ( translatedType );
        }

        return StructType::get (T->getContext(),
                                ArrayRef<Type *>(translatedTypes), 
                                T->isPacked() );
}

ArrayType * mapTypeToGlobal ( ArrayType * T )
{
        return T;
}

PointerType * mapTypeToGlobal ( PointerType * PT )
{
        Type * translatedType = mapTypeToGlobal ( PT->getElementType());
        return PointerType::get ( translatedType, GlobalAddressSpace );
}

SequentialType * mapTypeToGlobal ( SequentialType * T ) {
        ArrayType * AT = dyn_cast<ArrayType> (T);
        if ( AT ) return mapTypeToGlobal (AT);

        PointerType * PT = dyn_cast<PointerType> (T);
        if ( PT ) return mapTypeToGlobal (PT);
 
        return T;
}

CompositeType * mapTypeToGlobal (CompositeType * T)
{
        StructType * ST = dyn_cast<StructType> (T);
        if ( ST ) return mapTypeToGlobal ( ST );
       
        SequentialType * SQ = dyn_cast<SequentialType> (T);
        if ( SQ ) return mapTypeToGlobal ( SQ );

        DEBUG (llvm::errs () << "Unknown type "; T->dump(); );
        return T;
}

Type * mapTypeToGlobal (Type * T)
{
        CompositeType * C = dyn_cast<CompositeType>(T); 
        if ( !C ) return T;
        return mapTypeToGlobal (C);
}

/* Create a new function type based on the provided function so that
   each arguments that are pointers, or pointer types within composite
   types, are pointer to global */

FunctionType * createNewFunctionTypeWithPtrToGlobals (Function * F)
{
        FunctionType * baseType = F->getFunctionType();

        std::vector<Type *> translatedArgTypes;
/*
        for (unsigned arg = 0, arg_end = baseType->getNumParams();
             arg != arg_end; ++arg) {
                Type * argType = baseType->getParamType(arg);
                Type * translatedType = mapTypeToGlobal (argType);
                translatedArgTypes.push_back ( translatedType );
        }*/
        unsigned argIdx = 0;
        for (Function::arg_iterator A = F->arg_begin(), Ae = F->arg_end();
             A != Ae; ++A, ++argIdx) {
                Type * argType = baseType->getParamType(argIdx); 
                Type * translatedType;

                StringRef argName = A->getName();
                if (argName.equals("scratch")
                    || argName.equals("lds")
                    || argName.equals("scratch_count")) {
                        PointerType * Ptr = dyn_cast<PointerType>(argType);
                        assert(Ptr && "Pointer type expected");
                        translatedType = PointerType::get(Ptr->getElementType(),
                                                          LocalAddressSpace); 
                } else {
                        if (A->hasByValAttr()) {
                                PointerType * ptrType =
                                        cast<PointerType>(argType);
                                Type * elementType =
                                        ptrType->getElementType();
                                Type * translatedElement =
                                        mapTypeToGlobal(elementType);
                                translatedType = 
                                        PointerType::get(translatedElement,
                                                         0);
                        } else translatedType = mapTypeToGlobal (argType);
                }
                translatedArgTypes.push_back ( translatedType );
        }

        FunctionType * newType 
                = FunctionType::get(mapTypeToGlobal(baseType->getReturnType()), 
                                    ArrayRef<Type *>(translatedArgTypes),
                                    baseType->isVarArg());
        return newType;
}

void nameAndMapArgs (Function * newFunc, Function * oldFunc, ValueToValueMapTy& VMap)
{
        typedef Function::arg_iterator iterator;
        for (iterator old_arg = oldFunc->arg_begin(),
             new_arg = newFunc->arg_begin(),
             last_arg = oldFunc->arg_end();
             old_arg != last_arg; ++old_arg, ++new_arg) {
                VMap[old_arg] = new_arg;
                new_arg->setName(old_arg->getName());

        }
}

BasicBlock * getOrCreateEntryBlock (Function * F)
{
        if ( ! F->isDeclaration() ) return &F->getEntryBlock();
        return BasicBlock::Create(F->getContext(), "entry", F);
}

AllocaInst * createNewAlloca(Type * elementType,
                             AllocaInst* oldAlloca,
                             BasicBlock * dest)
{
        TerminatorInst * terminator = dest->getTerminator();
        if (terminator) {
                return new AllocaInst(elementType,
                                      oldAlloca->getArraySize(),
                                      oldAlloca->getName(),
                                      terminator);
        }
        return new AllocaInst(elementType, 
                              oldAlloca->getArraySize(),
                              oldAlloca->getName(),
                              dest);

}

void updateListWithUsers ( User *U, Value * oldOperand, Value * newOperand,
                           InstUpdateWorkList * updates )
{
    Instruction * Insn = dyn_cast<Instruction>(U);
    if ( Insn ) {
        updates->addUpdate (
                new ForwardUpdate(Insn,
                    oldOperand, newOperand ) );
    } else if (ConstantExpr * GEPCE =
            dyn_cast<ConstantExpr>(U)) {
        DEBUG(llvm::errs()<<"GEPCE:";
                GEPCE->dump(););
        // patch all the users of the constexpr by
        // first producing an equivalent instruction that
        // computes the constantexpr
        for(Value::use_iterator CU = GEPCE->use_begin(),
                CE = GEPCE->use_end(); CU!=CE;) {
            if (Instruction *I2 = dyn_cast<Instruction>(*CU)) {
                Insn = GEPCE->getAsInstruction();
                Insn->insertBefore(I2);
                updateInstructionWithNewOperand(Insn,
                        oldOperand, newOperand, updates);
                updateInstructionWithNewOperand(I2,
                        GEPCE, Insn, updates);
                // CU is invalidated
                CU = GEPCE->use_begin();
                continue;
            }
            CU++;
        }
    }
}
void updateListWithUsers ( Value::use_iterator U, const Value::use_iterator& Ue, 
                           Value * oldOperand, Value * newOperand,
                           InstUpdateWorkList * updates ) 
{
    for ( ; U != Ue; ++U ) {
        updateListWithUsers(*U, oldOperand, newOperand, updates);
    } 
}

void updateLoadInstWithNewOperand(LoadInst * I, Value * newOperand, InstUpdateWorkList * updatesNeeded)
{
        Type * originalLoadedType = I->getType();
        I->setOperand(0, newOperand);
        PointerType * PT = cast<PointerType>(newOperand->getType());
        if ( PT->getElementType() != originalLoadedType ) {
                I->mutateType(PT->getElementType());
                updateListWithUsers(I->use_begin(), I->use_end(), I, I, updatesNeeded);
        }
}

void updatePHINodeWithNewOperand(PHINode * I, Value * oldOperand,
        Value * newOperand, InstUpdateWorkList * updatesNeeded)
{
    // Update the PHI node itself as well its users
    Type * originalType = I->getType();
    PointerType * PT = cast<PointerType>(newOperand->getType());
    if ( PT != originalType ) {
        I->mutateType(PT);
        updateListWithUsers(I->use_begin(), I->use_end(), I, I, updatesNeeded);
    } else {
        return;
    }
    // Update the sources of the PHI node
    for (unsigned i = 0; i < I->getNumIncomingValues(); i++) {
        Value *V = I->getIncomingValue(i);
        if (V == oldOperand) {
            I->setOperand(i, newOperand);
        } else if (V == newOperand) {
            continue;
        } else if (!isa<Instruction>(V)) {
            // It is temping to do backward updating on other incoming
            // operands here, // but we don't have to. Eventually fwd
            // updates will cover them, except Undefs
            V->mutateType(PT);
        }
    }
}

void updateStoreInstWithNewOperand(StoreInst * I, Value * oldOperand, Value * newOperand, InstUpdateWorkList * updatesNeeded)
{
        unsigned index = I->getOperand(1) == oldOperand?1:0;
        I->setOperand(index, newOperand);

        Value * storeOperand = I->getPointerOperand();
        PointerType * destType =
                dyn_cast<PointerType>(storeOperand->getType());

        if ( destType->getElementType ()
             == I->getValueOperand()->getType() ) return;


        if ( index == StoreInst::getPointerOperandIndex () ) {
                DEBUG(llvm::errs() << "Source value should be updated\n";);
                DEBUG(llvm::errs() << " as "; I->getValueOperand()->dump();
                      llvm::errs() << " is stored in "; I->getPointerOperand()->dump(););
        } else {
                PointerType * newType = 
                        PointerType::get(I->getValueOperand()->getType(),
                                         destType->getAddressSpace());
                Instruction * ptrProducer = 
                        dyn_cast<Instruction> ( I->getPointerOperand () );

                BackwardUpdate::setExpectedType (ptrProducer,
                                                 newType, updatesNeeded);
                    
                                                 
        }
}

void updateCallInstWithNewOperand(CallInst * CI, Value * oldOperand, Value * newOperand, InstUpdateWorkList * updatesNeeded)
{
        for ( unsigned i = 0, numArgs = CI->getNumArgOperands();
              i != numArgs; ++i ) {
                if ( CI->getArgOperand ( i ) == oldOperand ) {
                        CI->setArgOperand ( i, newOperand );
                }
        }
}

void updateBitCastInstWithNewOperand(BitCastInst * BI, Value *oldOperand, Value * newOperand, InstUpdateWorkList * updatesNeeded)
{
        Type * currentType = BI->getType();
        PointerType * currentPtrType = dyn_cast<PointerType>(currentType);
        if (!currentPtrType) return;

        Type * sourceType = newOperand->getType();
        PointerType * sourcePtrType = dyn_cast<PointerType>(sourceType);
        if (!sourcePtrType) return;

        if ( sourcePtrType->getAddressSpace()
             == currentPtrType->getAddressSpace() ) return; 

        PointerType * newDestType = 
                PointerType::get(currentPtrType->getElementType(),
                                 sourcePtrType->getAddressSpace());

        BitCastInst * newBCI = new BitCastInst (newOperand, newDestType,
                                                "", BI);

        updateListWithUsers (BI->use_begin(), BI->use_end(),
                             BI, newBCI, updatesNeeded);
}

void updateGEPWithNewOperand(GetElementPtrInst * GEP, Value * oldOperand, Value * newOperand, InstUpdateWorkList * updatesNeeded)
{
        if ( GEP->getPointerOperand() != oldOperand ) return;

        std::vector<Value *> Indices(GEP->idx_begin(), GEP->idx_end());
       
        Type * futureType = 
                GEP->getGEPReturnType(newOperand, ArrayRef<Value *>(Indices)); 

        PointerType * futurePtrType = dyn_cast<PointerType>(futureType);
        if ( !futurePtrType ) return;

        GEP->setOperand ( GEP->getPointerOperandIndex(), newOperand);

        if ( futurePtrType == GEP->getType()) return;

        GEP->mutateType ( futurePtrType );
        updateListWithUsers(GEP->use_begin(), GEP->use_end(), GEP, GEP, updatesNeeded);
        
}

bool CheckCalledFunction ( CallInst * CI, InstUpdateWorkList * updates,
                           FunctionType *& newFunctionType )
{
        Function * CalledFunction = CI->getCalledFunction ();
        FunctionType * CalledType = CalledFunction->getFunctionType ();
        unsigned numParams = CalledType->getNumParams ();

        std::vector<Type *> newArgTypes;

        bool changeDetected = false;
        for ( unsigned param = 0; param < numParams; ++param ) {
                Type * paramType = CalledType->getParamType ( param );
                Value * argument = CI->getArgOperand ( param );
                Type * argType = argument->getType ();

                changeDetected |= ( paramType != argType );

                newArgTypes.push_back (argType);
        }

        if ( !changeDetected ) {
                return false;
        }

        Type * returnType = mapTypeToGlobal (CalledType->getReturnType());

        newFunctionType = 
                FunctionType::get(returnType,
                                  ArrayRef<Type *>(newArgTypes),
                                  CalledType->isVarArg());
        return true;
}

void promoteCallToNewFunction (CallInst * CI, FunctionType * newFunctionType,
                               InstUpdateWorkList * updates)
{
        Function * CalledFunction = CI->getCalledFunction ();
        Function * promoted = 
                createPromotedFunctionToType ( CalledFunction, newFunctionType);
        CI->setCalledFunction (promoted);

        Type * returnType = newFunctionType->getReturnType();
        if ( returnType == CI->getType () ) return;

        CI->mutateType (returnType);
        updateListWithUsers (CI->use_begin(), CI->use_end(),
                             CI, CI, updates);
}

void CollectChangedCalledFunctions (Function * F, InstUpdateWorkList * updatesNeeded)
{
        typedef std::vector<CallInst *> CallInstsTy;
        CallInstsTy foundCalls;
        for (Function::iterator B = F->begin(), Be = F->end();
             B != Be; ++B) {
                for (BasicBlock::iterator I = B->begin(), Ie = B->end();
                     I != Ie; ++I) {
                        CallInst * CI = dyn_cast<CallInst>(I);
                        if ( !CI ) continue;
                        foundCalls.push_back(CI);
                }
        }
        typedef CallInstsTy::iterator iterator;
        typedef std::pair<CallInst *, FunctionType *> PromotionTy;
        typedef std::vector<PromotionTy> ToPromoteTy ;
        ToPromoteTy changedFunctions;
        for (iterator C = foundCalls.begin(), Ce = foundCalls.end();
             C != Ce; ++C) {
                FunctionType * newType;
                if ( !CheckCalledFunction ( *C, updatesNeeded, newType ) )
                        continue;
                changedFunctions.push_back ( std::make_pair(*C, newType) );
        }

        for (ToPromoteTy::iterator C = changedFunctions.begin(),
             Ce = changedFunctions.end();
             C != Ce; ++C) {
                CallInst * CI = C->first;
                FunctionType *newType = C->second; 
                IntrinsicInst * Intrinsic = dyn_cast<IntrinsicInst>(CI);
                if (!Intrinsic) {
                        promoteCallToNewFunction(CI, newType, updatesNeeded);
                        continue;
                }
                Intrinsic::ID IntrinsicId = Intrinsic->getIntrinsicID ();
                ArrayRef<Type *> Args(newType->param_begin(),
                                      newType->param_begin()+3);
                Function * newIntrinsicDecl = 
                        Intrinsic::getDeclaration (F->getParent(),
                                                   IntrinsicId,
                                                   Args);
                DEBUG(llvm::errs() << "When updating intrinsic "; CI->dump(););
                CI->setCalledFunction (newIntrinsicDecl);
                DEBUG(llvm::errs() << " expecting: " 
                                   << Intrinsic::getName(IntrinsicId, Args);
                CI->dump();
                llvm::errs() << CI->getCalledFunction()->getName() << "\n";);
        }
}


                           
void updateInstructionWithNewOperand(Instruction * I, 
                                     Value * oldOperand, 
                                     Value * newOperand,
                                     InstUpdateWorkList * updatesNeeded)
{
       if (LoadInst * LI = dyn_cast<LoadInst>(I)) {
               updateLoadInstWithNewOperand(LI, newOperand, updatesNeeded);
               return;
       }
    
       if (StoreInst * SI = dyn_cast<StoreInst>(I)) {
               updateStoreInstWithNewOperand(SI, oldOperand, newOperand, updatesNeeded);
               return;
       }

       if (CallInst * CI = dyn_cast<CallInst>(I)) {
               updateCallInstWithNewOperand(CI, oldOperand, newOperand, updatesNeeded);
               return;
       }

       if (BitCastInst * BI = dyn_cast<BitCastInst>(I)) {
               updateBitCastInstWithNewOperand(BI, oldOperand, newOperand, updatesNeeded);
               return;
       }

       if (GetElementPtrInst * GEP = dyn_cast<GetElementPtrInst>(I)) {
               updateGEPWithNewOperand(GEP, oldOperand, newOperand, 
                                       updatesNeeded);
               return;
       }

       if (PHINode * PHI = dyn_cast<PHINode>(I)) {
           updatePHINodeWithNewOperand(PHI, oldOperand, newOperand, updatesNeeded);
           return;
       }

       DEBUG(llvm::errs() << "DO NOT KNOW HOW TO UPDATE INSTRUCTION: "; 
             I->print(llvm::errs()); llvm::errs() << "\n";);
}  

// tile_static are declared as static variables in section("clamp_opencl_local")
// for each tile_static, make a modified clone with address space 3 and update users
void promoteTileStatic(Function *Func, InstUpdateWorkList * updateNeeded)
{
    Module *M = Func->getParent();
    Module::GlobalListType &globals = M->getGlobalList();
    for (Module::global_iterator I = globals.begin(), E = globals.end();
        I != E; I++) {
        if (!I->hasSection() || 
            I->getSection() != std::string(TILE_STATIC_NAME) ||
            I->getType()->getPointerAddressSpace() != 0 ||
            !I->hasName()) {
            continue;
        }
        DEBUG(llvm::errs() << "Promoting variable\n";
                I->dump(););
        std::set<Function *> users;
        typedef std::multimap<Function *, llvm::User *> Uses;
        Uses uses;
        for (Value::use_iterator U = I->use_begin(), Ue = I->use_end();
            U!=Ue;) {
            if (Instruction *Ins = dyn_cast<Instruction>(*U)) {
                users.insert(Ins->getParent()->getParent());
                uses.insert(std::make_pair(Ins->getParent()->getParent(), *U));
            } else if (ConstantExpr *C = dyn_cast<ConstantExpr>(*U)) {
                // Replace GEPCE uses so that we have an instruction to track
                updateListWithUsers (*U, I, I, updateNeeded);
                assert(U->getNumUses() == 0);
                C->destroyConstant();
                U = I->use_begin();
                continue;
            }
            DEBUG(llvm::errs() << "U: \n";
                U->dump(););
            U++;
        }
        int i = users.size()-1;
        // Create a clone of the tile static variable for each unique
        // function that uses it
        for (std::set<Function*>::reverse_iterator
                F = users.rbegin(), Fe = users.rend();
                F != Fe; F++, i--) {
            GlobalVariable *new_GV = new GlobalVariable(*M,
                    I->getType()->getElementType(),
                    I->isConstant(), I->getLinkage(),
                    I->hasInitializer()?I->getInitializer():0,
                    "", (GlobalVariable *)0, I->getThreadLocalMode(), LocalAddressSpace);
            new_GV->copyAttributesFrom(I);
            if (i == 0) {
                new_GV->takeName(I);
            } else {
                new_GV->setName(I->getName());
            }
            std::pair<Uses::iterator, Uses::iterator> usesOfSameFunction;
            usesOfSameFunction = uses.equal_range(*F);
            for ( Uses::iterator U = usesOfSameFunction.first, Ue =
                usesOfSameFunction.second; U != Ue; U++) 
                updateListWithUsers (U->second, I, new_GV, updateNeeded);
        }
    }
}

void eraseOldTileStaticDefs(Module *M)
{
    std::vector<GlobalValue*> todo;
    Module::GlobalListType &globals = M->getGlobalList();
    for (Module::global_iterator I = globals.begin(), E = globals.end();
        I != E; I++) {
        if (!I->hasSection() || 
            I->getSection() != std::string(TILE_STATIC_NAME) ||
            I->getType()->getPointerAddressSpace() != 0) {
            continue;
        }
        I->removeDeadConstantUsers();
        if (I->getNumUses() == 0)
            todo.push_back(I);
    }
    for (std::vector<GlobalValue*>::iterator I = todo.begin(),
            E = todo.end(); I!=E; I++) {
        (*I)->eraseFromParent();
    }
}

void promoteAllocas (Function * Func,  
                     InstUpdateWorkList * updatesNeeded)
{
        typedef BasicBlock::iterator iterator;
        for (iterator I = Func->begin()->begin();
             isa<AllocaInst>(I); ++I) {
                AllocaInst * AI = cast<AllocaInst>(I);
                Type * allocatedType = AI->getType()->getElementType();
                Type * promotedType = mapTypeToGlobal(allocatedType);
                
                if ( allocatedType == promotedType ) continue;

                AllocaInst * clonedAlloca = new AllocaInst(promotedType,
                                                           AI->getArraySize(),
                                                           "", AI);

                updateListWithUsers ( AI->use_begin(), AI->use_end(), 
                                      AI, clonedAlloca, updatesNeeded );
        } 
}

void promoteBitcasts (Function * F, InstUpdateWorkList * updates)
{
        typedef std::vector<BitCastInst *> BitCastList;
        BitCastList foundBitCasts;
        for (Function::iterator B = F->begin(), Be = F->end();
             B != Be; ++B) {
                for (BasicBlock::iterator I = B->begin(), Ie = B->end();
                     I != Ie; ++I) {
                        BitCastInst * BI = dyn_cast<BitCastInst>(I);
                        if ( ! BI ) continue;
                        foundBitCasts.push_back(BI);
                }
        }

        for (BitCastList::const_iterator I = foundBitCasts.begin(),
             Ie = foundBitCasts.end(); I != Ie; ++I) {
                BitCastInst * BI = *I;  

                Type *destType = BI->getType();
                PointerType * destPtrType = 
                        dyn_cast<PointerType>(destType);
                if ( ! destPtrType ) continue;

                Type * srcType = BI->getOperand(0)->getType();
                PointerType * srcPtrType =
                        dyn_cast<PointerType>(srcType);
                if ( ! srcPtrType ) continue;
#if 0
                unsigned srcAddressSpace = 
                        srcPtrType->getAddressSpace();

                unsigned destAddressSpace = 
                        destPtrType->getAddressSpace();
#endif
                Type * elementType = destPtrType->getElementType();
                Type * mappedType = mapTypeToGlobal(elementType);
                unsigned addrSpace = srcPtrType->getAddressSpace();
                Type * newDestType = PointerType::get(mappedType, addrSpace);
                if (elementType == mappedType) continue;

                        
                BitCastInst * newBI = new BitCastInst(BI->getOperand(0),
                                                      newDestType, BI->getName(),
                                                         BI);
                updateListWithUsers (BI->use_begin(), BI->use_end(),
                                     BI, newBI, updates);
        }

}

bool hasPtrToNonZeroAddrSpace (Value * V)
{
        Type * ValueType = V->getType();
        PointerType * ptrType = dyn_cast<PointerType>(ValueType);
        if ( !ptrType ) return false;
        return true;
        return ptrType->getAddressSpace() != 0;

}

void updateArgUsers (Function * F, InstUpdateWorkList * updateNeeded)
{
        typedef Function::arg_iterator arg_iterator;
        for (arg_iterator A = F->arg_begin(), Ae = F->arg_end();
             A != Ae; ++A) {
                if ( !hasPtrToNonZeroAddrSpace (A) ) continue;
                updateListWithUsers (A->use_begin(), A->use_end(),
                                     A, A, updateNeeded); 
        }
}


// updateOperandType - Replace types of operand and return values with the promoted types if necessary
// This function goes through the function's body and handles GEPs and select instruction specially.
// After a function is cloned by calling CloneFunctionInto, some of the operands types 
// might not be updated correctly. Neither are some of  the instructions' return types.
// For example, 
// (1) getelementptr instruction will leave type of its pointer operand un-promoted 
// (2) select instructiono will not update its return type as what has been changed to its #1 or #2 operand
// Note that It is always safe to call this function right after CloneFunctionInto
//
void updateOperandType(Function * oldF, Function * newF, FunctionType* ty, InstUpdateWorkList* workList)
{
  // Walk all the BBs
  for (Function::iterator B = newF->begin(), Be = newF->end();B != Be; ++B) {
    // Walk all instructions	
    for (BasicBlock::iterator I = B->begin(), Ie = B->end(); I != Ie; ++I) {
      if (SelectInst *Sel = dyn_cast<SelectInst>(I)) {
        assert(Sel->getOperand(1) && "#1  operand  of Select Instruction is invalid!");
        Sel->mutateType(I->getOperand(1)->getType());
        updateListWithUsers(I->use_begin(), I->use_end(), I, I, workList);
      } else if( GetElementPtrInst *GEP = dyn_cast<GetElementPtrInst>(I)) {
        // Handle GEPs
        Type*T = GEP->getPointerOperandType();
        // Traverse the old args to find source type
        unsigned argIdx = 0;
        for (Function::arg_iterator A = oldF->arg_begin(), Ae = oldF->arg_end();
                    A != Ae; ++A, ++argIdx) {
          // Since Type* is immutable, pointer comparison to see if they are the same
          if(T == oldF->getFunctionType()->getParamType(argIdx)) {
            Argument* V = new Argument(ty->getParamType(argIdx), GEP->getPointerOperand()->getName());
          // Note that only forward udpate is allowed. A backward update
          updateGEPWithNewOperand(GEP, GEP->getPointerOperand(), V, workList);
          }
        }
      }
    }
  }
}

Function * createPromotedFunctionToType ( Function * F, FunctionType * promoteType)
{
        DEBUG(llvm::errs() << "========================================\n";);
        Function * newFunction = Function::Create (promoteType,
                                                   F->getLinkage(),
                                                   F->getName(),
                                                   F->getParent());
        DEBUG(llvm::errs() << "New function name: " << newFunction->getName() << "\n" << "\n";);

        ValueToValueMapTy CloneMapping;
        nameAndMapArgs(newFunction, F, CloneMapping);    


        SmallVector<ReturnInst *, 1> Returns;
        CloneFunctionInto(newFunction, F, CloneMapping, false, Returns);

        ValueToValueMapTy CorrectedMapping;
        InstUpdateWorkList workList;
//        promoteAllocas(newFunction, workList);
//        promoteBitcasts(newFunction, workList);
        promoteTileStatic(newFunction, &workList);
        updateArgUsers (newFunction, &workList);
        updateOperandType(F, newFunction, promoteType, &workList);
       
        do {
                /*while( !workList.empty() ) {
                        update_token update = workList.back();
                        workList.pop_back();
                        updateInstructionWithNewOperand (update.subject,
                                                         update.oldOperand,
                                                         update.newOperand,
                                                         workList); 

                }*/
                workList.run();
                CollectChangedCalledFunctions ( newFunction, &workList );
        } while ( !workList.empty() );
        
        eraseOldTileStaticDefs(F->getParent());
        if (verifyFunction (*newFunction, PrintMessageAction)) {
                llvm::errs() << "When checking the updated function of: ";
                F->dump();
                llvm::errs() << " into: ";
                newFunction->dump(); 
        }
        DEBUG(llvm::errs() << "-------------------------------------------";);
        return newFunction;
}

Function * createPromotedFunction ( Function * F )
{
        FunctionType * promotedType = 
                createNewFunctionTypeWithPtrToGlobals (F);
        return createPromotedFunctionToType (F, promotedType);
}

/* A visitor function which is called for each MDNode
   located into another MDNode */
class KernelNodeVisitor {
        public:
        KernelNodeVisitor(FunctionVect& FV);
        void operator()(MDNode * N);

        private:
        FunctionVect& found_kernels;
};

KernelNodeVisitor::KernelNodeVisitor(FunctionVect& FV)
        : found_kernels(FV)
{}

void KernelNodeVisitor::operator()(MDNode *N)
{
        if ( N->getNumOperands() < 1) return;
        Value * Op = N->getOperand(0);
        if (!Op)
            return;
        if ( Function * F = dyn_cast<Function>(Op)) {
                found_kernels.push_back(F); 
        }
}

/* Call functor for each MDNode located within the Named MDNode */
void visitMDNodeOperands(NamedMDNode * N, KernelNodeVisitor& visitor)
{
        for (unsigned operand = 0, end = N->getNumOperands();
             operand < end; ++operand) {
                visitor(N->getOperand(operand));
        }
}

/* Accumulate LLVM functions that are kernels within the
   found_kernels vector. Return true if kernels are found.
   False otherwise. */
bool findKernels(Module& M, FunctionVect& found_kernels)
{
        NamedMDNode * root = getKernelListMDNode(M);
        if (!root || (root->getNumOperands() == 0)) return false;

        KernelNodeVisitor visitor(found_kernels);
        visitMDNodeOperands(root, visitor);

        return found_kernels.size() != 0;
}

void updateKernels(Module& M, const FunctionMap& new_kernels)
{
        NamedMDNode * root = getKernelListMDNode(M);
        typedef FunctionMap::const_iterator iterator;
        // for each kernel..
        for (unsigned i = 0; i < root->getNumOperands(); i++) {
            // for each metadata of the kernel..
            MDNode * kernel = root->getOperand(i);
            Function * f = dyn_cast<Function>(kernel->getOperand(0));
            assert(f != NULL);
            iterator I = new_kernels.find(f);
            if (I != new_kernels.end())
                kernel->replaceOperandWith(0, I->second);
        }
        for (iterator kern = new_kernels.begin(), end = new_kernels.end();
             kern != end; ++kern) {
                // Remove the original function
                kern->first->deleteBody();
                kern->first->setCallingConv(llvm::CallingConv::C);
        }
}

StructType * wrapStructType (StructType * src, Type * Subst) {
        std::vector<Type *> ElementTypes;
        bool changed = false;
        typedef StructType::element_iterator iterator;

        for (iterator E = src->element_begin(), Ee = src->element_end();
             E != Ee; ++E) {
                Type * newType = *E;
                PointerType * ptrType = dyn_cast<PointerType>(newType);
                if (ptrType) {
                        newType = Subst;
                        changed = true;
                }
                ElementTypes.push_back(newType);
        }

        if (!changed) return src;

        return StructType::get (src->getContext(),
                                ArrayRef<Type *>(ElementTypes),
                                src->isPacked());
}

Function * createWrappedFunction (Function * F)
{
        typedef Function::arg_iterator arg_iterator;
        typedef std::vector<Type *> WrappedTypes;
        typedef std::pair<unsigned, StructType *> WrapPair;
        typedef std::vector<WrapPair> WrappingTodo;
        WrappedTypes wrappedTypes;
        WrappingTodo Todo;
        bool changed = false;
        unsigned argNum  = 0;
        Module * M = F->getParent ();
        DataLayout TD(M);
        unsigned ptrSize = TD.getPointerSizeInBits ();
        Type * PtrDiff = Type::getIntNTy (F->getContext(), ptrSize);
        for (arg_iterator A = F->arg_begin(), Ae = F->arg_end();
             A != Ae; ++A, ++argNum) {
                Type * argType = A->getType ();
                if (!A->hasByValAttr()) {
                        wrappedTypes.push_back (argType);
                        continue;
                }
                // ByVal args are pointers
                PointerType * ptrArgType = cast<PointerType>(argType);
                StructType * argStructType = 
                        dyn_cast<StructType>(ptrArgType->getElementType());
                if (!argStructType) {
                        wrappedTypes.push_back (argType);
                        continue;
                }
                StructType * wrapped = 
                        wrapStructType (argStructType, PtrDiff);
                if (wrapped == argStructType) {
                        wrappedTypes.push_back (argType);
                        continue;
                }
                PointerType * final = 
                        PointerType::get(wrapped,
                                         ptrArgType->getAddressSpace());
                wrappedTypes.push_back(final);
                Todo.push_back (std::make_pair(argNum, argStructType));
                changed = true;
        }
        if ( !changed ) return F;
        
        FunctionType * newFuncType =
                FunctionType::get(F->getReturnType(),
                                  ArrayRef<Type*>(wrappedTypes),
                                  F->isVarArg());
        Function * wrapped = Function::Create (newFuncType,
                                               F->getLinkage(),
                                               F->getName(),
                                               M);
        std::vector<Value *> callArgs;
        for (arg_iterator sA = F->arg_begin(), dA = wrapped->arg_begin(),
                          Ae = F->arg_end(); sA != Ae; ++sA, ++dA) {
            dA->setName (sA->getName());
            callArgs.push_back(dA);
        }
        wrapped->setAttributes (F->getAttributes());
        BasicBlock * entry = BasicBlock::Create(F->getContext(), "entry",
                                                wrapped, NULL);

        Type * BoolTy = Type::getInt1Ty(F->getContext());
        Type * Int8Ty = Type::getInt8Ty(F->getContext());
        Type * Int32Ty = Type::getInt32Ty(F->getContext());
        Type * Int64Ty = Type::getInt64Ty(F->getContext());
        Type * castSrcType = PointerType::get(Int8Ty, 0);
        Type * castTargetType = PointerType::get(Int8Ty, 0);

        std::vector<Type *> MemCpyTypes;
        MemCpyTypes.push_back (castTargetType);
        MemCpyTypes.push_back (castSrcType);
        MemCpyTypes.push_back (Int64Ty);
        Function * memCpy = Intrinsic::getDeclaration (M, Intrinsic::memcpy,
                                                       ArrayRef<Type *>(MemCpyTypes));
        Constant * align = ConstantInt::get (Int32Ty, 4, false);
        Constant * isVolatile = ConstantInt::getFalse (BoolTy);
        for (WrappingTodo::iterator W = Todo.begin(), We = Todo.end();
             W != We; ++W) {
                Function::arg_iterator A = wrapped->arg_begin();
                for (unsigned i = 0; i < W->first; ++i, ++A) {}
                AllocaInst * AI = new AllocaInst(W->second, NULL, "", entry);
                std::vector<Value *> memCpyArgs;
                memCpyArgs.push_back (new BitCastInst (AI, castTargetType,
                                                       "", entry));
                memCpyArgs.push_back (new BitCastInst (A, castSrcType,
                                                       "", entry));

                memCpyArgs.push_back (ConstantInt::get(Int64Ty,
                                                       TD.getTypeStoreSize(W->second),
                                                       false));
                memCpyArgs.push_back (align);
                memCpyArgs.push_back (isVolatile);
                CallInst::Create (memCpy, ArrayRef<Value *>(memCpyArgs),
                                  "", entry);
                callArgs [W->first] = AI;
        }

        CallInst::Create (F, ArrayRef <Value *> (callArgs), "", entry);
        ReturnInst::Create (F->getContext(), NULL, entry);
        return wrapped;
}


class PromoteGlobals : public ModulePass {
        public:
        static char ID;
        PromoteGlobals();
        virtual ~PromoteGlobals();
        virtual void getAnalysisUsage(AnalysisUsage& AU) const;
        bool runOnModule(Module& M);
};
} // ::<unnamed> namespace

PromoteGlobals::PromoteGlobals() : ModulePass(ID)
{}

PromoteGlobals::~PromoteGlobals()
{}

void PromoteGlobals::getAnalysisUsage(AnalysisUsage& AU) const
{
        AU.addRequired<CallGraph>();
}
static std::string escapeName(const std::string &orig_name)
{
    std::string oldName(orig_name);
    // AMD OpenCL doesn't like kernel names starting with _
    if (oldName[0] == '_')
        oldName = oldName.substr(1);
    size_t loc;
    // escape name: $ -> _EC_
    while ((loc = oldName.find('$')) != std::string::npos) {
        oldName.replace(loc, 1, "_EC_");
    }
    return oldName;
}

bool PromoteGlobals::runOnModule(Module& M) 
{
        FunctionVect foundKernels;
        FunctionMap promotedKernels;
        if (!findKernels(M, foundKernels)) return false;

        typedef FunctionVect::const_iterator kernel_iterator;
        for (kernel_iterator F = foundKernels.begin(), Fe = foundKernels.end();
             F != Fe; ++F) {
                if ((*F)->empty())
                    continue;
                Function * promoted = createPromotedFunction (*F);
                promoted->takeName (*F);
                promoted->setName(escapeName(promoted->getName().str()));

                promoted->setCallingConv(llvm::CallingConv::SPIR_KERNEL);
                // lambdas can be set as internal. This causes problem
                // in optimizer and we shall mark it as non-internal
                if (promoted->getLinkage() ==
                        GlobalValue::InternalLinkage) {
                    promoted->setLinkage(GlobalValue::ExternalLinkage);
                }
                (*F)->setLinkage(GlobalValue::InternalLinkage);
                promotedKernels[*F] = promoted;
        }
        updateKernels (M, promotedKernels);

        // If the barrier present is used, we need to ensure it cannot be duplicated.
        for (Module::iterator F = M.begin(), Fe = M.end(); F != Fe; ++F) {
                StringRef name = F->getName();
                if (name.equals ("barrier")) {
                        F->addFnAttr (Attribute::NoDuplicate);
                }
        }

        // Rename local variables per SPIR naming rule
        Module::GlobalListType &globals = M.getGlobalList();
        for (Module::global_iterator I = globals.begin(), E = globals.end();
                I != E; I++) {
            if (I->hasSection() && 
                    I->getSection() == std::string(TILE_STATIC_NAME) &&
                    I->getType()->getPointerAddressSpace() != 0) {
                std::string oldName = escapeName(I->getName().str());
                // Prepend the name of the function which contains the user
                std::set<std::string> userNames;
                for (Value::use_iterator U = I->use_begin(), Ue = I->use_end();
                    U != Ue; U ++) {
                    Instruction *Ins = dyn_cast<Instruction>(*U);
                    if (!Ins)
                        continue;
                    userNames.insert(Ins->getParent()->getParent()->getName().str());
                }
                // A local memory variable belongs to only one kernel, per SPIR spec
                assert(userNames.size() < 2 &&
                        "__local variable belongs to more than one kernel");
                if (userNames.empty())
                    continue;
                oldName = *(userNames.begin()) + "."+oldName;
                I->setName(oldName);
                // AMD SPIR stack takes only internal linkage
                if (I->hasInitializer())
                    I->setLinkage(GlobalValue::InternalLinkage);
            }
        }
        return false;
}


char PromoteGlobals::ID = 0;
#if 1
static RegisterPass<PromoteGlobals>
Y("promote-globals", "Promote Pointer To Global Pass");
#else
INITIALIZE_PASS(PromoteGlobals, "promote-globals", "Promote Pointer to Global", false, false);
#endif // BoltTranslator_EXPORTS

llvm::ModulePass * createPromoteGlobalsPass ()
{
        return new PromoteGlobals;
}
