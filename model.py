import torch
from torch import nn
from embeddingupdation import EmbeddingUpdation,RelationUpdateLayers,ClassUpdateLayer,MLPclasses

class DeepOntologicalNetwork(nn.Module):
    def __init__(self,no_of_individuals,emb_dim,relation_names,ind_class_membership_mat,no_of_classes):
        super().__init__()
        self.emb_mat = EmbeddingUpdation(no_of_individuals,emb_dim)
        self.relationupdatelayers = RelationUpdateLayers(relation_names,emb_dim,self.emb_mat)
        self.classupdatelayer = ClassUpdateLayer(emb_dim,no_of_classes,no_of_individuals,self.emb_mat,ind_class_membership_mat)
        self.classpredictionlayer = MLPclasses(emb_dim,no_of_classes)

    def forward(self,emb_iterations):
     
        sub_index = 0
        obj_index = 1

        subject_emb = self.emb_mat.get_emb(sub_index)
        object_emb = self.emb_mat.get_emb(obj_index)
        for i in range(emb_iterations):
            self.relationupdatelayers.forward("ParentOf",False, "subject_",0,subject_emb,object_emb)
          
            self.relationupdatelayers.forward("ParentOf",False,"object_",1,subject_emb,object_emb)
            

            self.classupdatelayer.forward()
        
        output = self.classpredictionlayer.forward(self.emb_mat.embeddings)

        return output
    


individuals_vocab = {"a":0,"b":1,"c":2}
no_of_individuals = len(individuals_vocab)
emb_dim = 3
relation_names = {"ParentOf", "WifeOf"}
class_names = {"Human","Male","Female"}
no_of_classes = 4

ind_class_membership_mat = torch.randint(-1,2,(no_of_individuals,no_of_classes))



model = DeepOntologicalNetwork(no_of_individuals,emb_dim,relation_names,ind_class_membership_mat,no_of_classes)



emb_iterations = 2
epochs = 10


criterion = nn.BCELoss()
ground_truth = torch.randint(0,2,(no_of_individuals,no_of_classes),dtype=torch.float32)

optimizers = torch.optim.SGD(model.parameters(),lr  = 0.01)
for epoch in range(epochs):
    out = model.forward(emb_iterations)

    loss = criterion(out, ground_truth)
    print("loss: ",loss)

    loss.backward(retain_graph = True)
    optimizers.step()


