from pyspark.ml.classification import RandomForestClassificationModel
from pyspark.sql import SparkSession





spark = (SparkSession.builder.appName("trainmodel_phanlop")
         .config("spark.hadoop.fs.defaultFS", "hdfs://localhost:9000")
         .getOrCreate())
# Tải lại mô hình Random Forest từ HDFS
loaded_rf_model = RandomForestClassificationModel.load("hdfs://localhost:9000/thanhtin/rf_model")

# Đọc tập dữ liệu test từ HDFS
test_path = "hdfs://localhost:9000/thanhtin/test-airline.csv"
test_df = spark.read.csv(test_path, header=True, inferSchema=True)

# Loại bỏ các giá trị null nếu có
test_df = test_df.na.drop()

# Mã hóa và xử lý dữ liệu test như đã làm với tập train
from pyspark.ml.feature import StringIndexer, VectorAssembler

# Mã hóa các cột chuỗi (lưu ý: sử dụng mô hình indexer đã fit từ trước nếu có thể)
indexers = [
    StringIndexer(inputCol=column, outputCol=column + "_index").fit(test_df)
    for column in ["Gender", "Customer Type", "Type of Travel", "Class"]
]

for indexer in indexers:
    test_df = indexer.transform(test_df)

# Danh sách các cột đặc trưng đã mã hóa
feature_columns = ["Age", "Flight Distance", "Inflight wifi service", "Departure/Arrival time convenient",
                   "Ease of Online booking", "Gate location", "Food and drink", "Online boarding",
                   "Seat comfort", "Inflight entertainment", "On-board service", "Leg room service",
                   "Baggage handling", "Checkin service", "Inflight service", "Cleanliness",
                   "Departure Delay in Minutes", "Arrival Delay in Minutes",
                   "Gender_index", "Customer Type_index", "Type of Travel_index", "Class_index"]

# Tạo cột 'features' với VectorAssembler
assembler = VectorAssembler(inputCols=feature_columns, outputCol="features")
test_df = assembler.transform(test_df)

# Mã hóa cột 'satisfaction' thành 'label' (nếu cần thiết)
satisfaction_indexer = StringIndexer(inputCol="satisfaction", outputCol="label").fit(test_df)
test_df = satisfaction_indexer.transform(test_df)

# Dự đoán trên tập test với mô hình đã tải
test_predictions = loaded_rf_model.transform(test_df)

# Hiển thị một vài kết quả dự đoán
test_predictions.select("features", "label", "prediction").show(5)
spark.stop()