import numpy
from sklearn.svm.classes import SVC
import Features_manager
import Database_manager
from sklearn.metrics.classification import precision_recall_fscore_support, accuracy_score

#initializate database_manager
database_manager=Database_manager.make_database_manager()
#initializate feature_manager
feature_manager=Features_manager.make_feature_manager()


tweets_training=numpy.array(database_manager.return_tweets_training())
labels_training=numpy.array(feature_manager.get_label(tweets_training))

tweets_test=numpy.array(database_manager.return_tweets_test())
labels_test=numpy.array(feature_manager.get_label(tweets_test))

#feature_type=feature_manager.get_availablefeaturetypes()

feature_type=[
            "numhashtag",
            "puntuactionmarks",
            "length",
            ]



X,X_test,feature_name,feature_index=feature_manager.create_feature_space(tweets_training,feature_type,tweets_test)


print(feature_name)
print("feature space dimension X:", X.shape)
print("feature space dimension X_test:", X_test.shape)


clf = SVC(kernel="linear")

clf.fit(X,labels_training)
test_predict = clf.predict(X_test)



prec, recall, f, support = precision_recall_fscore_support(
labels_test,
test_predict,
beta=1)

accuracy = accuracy_score(
test_predict,
labels_test
)

print(prec, recall, f, support )
print(accuracy)


