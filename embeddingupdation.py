import torch
from torch import nn


#these are for predictions
class MLPclasses(nn.Module):
    def __init__(self, emb_dim, no_of_classes):
        super().__init__()
        self.input_dim = emb_dim
        self.no_of_classes = no_of_classes
        self.ff1 = nn.Linear(self.input_dim,8)
        self.relu = nn.ReLU()
        self.ff2 = nn.Linear(8,9)
        self.outputlayer = nn.Linear(9,self.no_of_classes)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):                               #x = emb_mat [noofind,embdim]
        x = self.ff1(x)
        x = self.relu(x)
        x = self.ff2(x)
        x = self.relu(x)
        x = self.outputlayer(x)
        x = self.sigmoid(x)
        return x
    


class UpdateLayerRelation(nn.Module):
    def __init__(self,emb_dim):
        super().__init__()
        self.es_gate_parameter = torch.nn.Parameter(torch.empty([emb_dim,emb_dim]))
        nn.init.kaiming_uniform_(self.es_gate_parameter)
        self.eo_gate_parameter = torch.nn.Parameter(torch.empty([emb_dim,emb_dim]))
        nn.init.kaiming_uniform_(self.eo_gate_parameter)
        
        self.es_parameter = torch.nn.Parameter(torch.empty([emb_dim,emb_dim]))
        nn.init.kaiming_uniform_(self.es_parameter)
        self.eo_parameter = torch.nn.Parameter(torch.empty([emb_dim,emb_dim]))
        nn.init.kaiming_uniform_(self.eo_parameter)

        self.es_eo_parameter = torch.nn.Parameter(torch.empty(emb_dim,1))
        nn.init.kaiming_uniform_(self.es_eo_parameter)

        self.relu = nn.ReLU()

        self.sigmoid = nn.Sigmoid()



    def forward(self,subject_emb, object_emb,sub_or_obj):
        gate = torch.matmul(self.es_gate_parameter,subject_emb) + torch.matmul(self.eo_gate_parameter,object_emb)
        gate = self.sigmoid(gate)

        y = torch.matmul(self.es_parameter,subject_emb) + torch.matmul(self.eo_parameter,object_emb)
        z = torch.matmul(subject_emb, object_emb.T)
        z = torch.matmul(z, self.es_eo_parameter)
        y = y+z
        y = self.relu(y)

        
        if sub_or_obj=="subject_":
            final = torch.mul(gate,y)
            final = final + subject_emb
        elif sub_or_obj == "object_":
            final = torch.mul(gate,y)
            final = final +object_emb
        final = torch.nn.functional.normalize(final, dim =0)
        return final


class RelationUpdateLayers(nn.Module):
    def __init__(self,relation_names,emb_dim,emb_matrix_class):
        super().__init__()
        self.all_relationlayers = nn.ModuleDict()
        self.emb_matrix = emb_matrix_class
        for relation_name in relation_names:

            subject_relation_name = "subject_"+relation_name
            self.all_relationlayers[subject_relation_name] = UpdateLayerRelation(emb_dim)

            object_relation_name = "object_"+relation_name
            self.all_relationlayers[object_relation_name] = UpdateLayerRelation(emb_dim)

            negation_sub_relation_name = "negation_subject_" + relation_name
            self.all_relationlayers[negation_sub_relation_name] = UpdateLayerRelation(emb_dim)

            negation_obj_relation_name = "negation_object_" + relation_name
            self.all_relationlayers[negation_obj_relation_name] = UpdateLayerRelation(emb_dim)


        
    def forward(self,relation_name,negation,sub_or_obj,emb_index,subject_emb, object_emb):
        if negation==True:
            operation = "negation_"+sub_or_obj
        else:
            operation = sub_or_obj
        layer_name = operation + relation_name
        relation_layer = self.all_relationlayers[layer_name]
        update_emb = relation_layer(subject_emb,object_emb, sub_or_obj)

        self.emb_matrix.update_emb(emb_index,update_emb)


#this is dummy for now
def get_class_memberships(subject_index,no_of_classes):
    return torch.tensor([1,0,0,0,1,1,-1,1,-1],dtype=float,requires_grad=False).view(no_of_classes,1)

class ClassUpdateLayer(nn.Module):
    def __init__(self,emb_dim,no_of_classes,no_of_individuals,emb_mat,ind_class_membership_matrix):    #ind_class_membership_matrix = [no_of_individuals, no_of_classes]
        super().__init__()
        self.no_of_classes = no_of_classes
        self.no_of_individuals = no_of_individuals 
        self.emb_dim = emb_dim
        self.ind_class_membership_matrix = ind_class_membership_matrix
        self.emb_mat = emb_mat
        self.V_parameter = torch.nn.Parameter(torch.empty((emb_dim, emb_dim+no_of_classes)))
        nn.init.kaiming_uniform_(self.V_parameter)

        self.W_parameter = torch.nn.Parameter(torch.empty((emb_dim,emb_dim+no_of_classes)))
        nn.init.kaiming_uniform_(self.W_parameter)

        self.sigmoid = nn.Sigmoid()
        self.relu = nn.ReLU()

    
    def forward(self):
        emb = self.emb_mat.embeddings        #[no_of_individuals,emb_dim]
        class_memberships = self.ind_class_membership_matrix      #expected : [no_of_individuals, no_of_classes]
        new_mat = torch.cat((emb,class_memberships),dim=1).view(self.emb_dim+self.no_of_classes,self.no_of_individuals)

        gate = torch.matmul(self.V_parameter,new_mat)
        gate = self.sigmoid(gate)

        e1 = torch.matmul(self.W_parameter,new_mat)
        e1 = self.relu(e1)

        final = torch.mul(e1,gate)     #[emb_dim,no_of_ind]
        final = final.view(self.no_of_individuals,self.emb_dim)
        final = emb + final
        final = torch.nn.functional.normalize(final,dim= 1)  #[no_of_individuals,emb_dim] - about no_of_individuals
        
        self.emb_mat.update_matrix(final)


#assuming no embedding matrix given
class EmbeddingUpdation(nn.Module):
    def __init__(self,no_of_individuals,emb_dim):
        super().__init__()
        self.emb_dim = emb_dim
        self.no_of_individuals = no_of_individuals
        self.embeddings = torch.empty((no_of_individuals,emb_dim),requires_grad=False)
        with torch.no_grad():
            self.embeddings.uniform_(-1,1)
            self.embeddings.data = torch.nn.functional.normalize(self.embeddings.data,dim=1)



    def forward(self,emb_index,new_emb):     #we use this method to update layers  
        mask = torch.zeros((1,self.emb_dim))
        print("new:", new_emb)
        print("embeddings:",self.embeddings[emb_index,:].view(1,self.emb_dim))
        emb = self.embeddings[emb_index,:].view(1,self.emb_dim)
        emb = emb*mask
        
        print(emb.size() ,"+",new_emb.size())
        emb = emb + new_emb.T
        print(emb.size)
        new_embeddings = self.embeddings.clone()
        print(new_embeddings.size())
        new_embeddings[emb_index,:] = emb
         
        print(new_embeddings)
        self.embeddings = new_embeddings


    def update_emb(self,emb_index,new_emb):
        self.forward(emb_index,new_emb)

    def get_emb(self, emb_index):
        return self.embeddings[emb_index,:].view(self.emb_dim,1)

    def update_matrix(self,new_matrix):
        mask = torch.zeros((self.no_of_individuals,self.emb_dim))
        self.embeddings = torch.mul(self.embeddings,mask)
        self.embeddings = self.embeddings + new_matrix


