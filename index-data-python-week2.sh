echo "Creating index settings and mappings"
curl -k -X DELETE -u admin:admin  "https://localhost:9200/bbuy_products"
curl -k -X DELETE -u admin:admin  "https://localhost:9200/bbuy_queries"

echo "Indexing with mappings from folder week2/conf"
curl -k -X PUT -u admin:admin  "https://localhost:9200/bbuy_products" -H 'Content-Type: application/json' -d @week2/conf/bbuy_products.json
curl -k -X PUT -u admin:admin  "https://localhost:9200/bbuy_queries" -H 'Content-Type: application/json' -d @week2/conf/bbuy_queries.json

echo "Indexing product data"
nohup python index_products.py -s workspace/datasets/product_data/products > workspace/logs/index_products.log &

echo "Indexing queries data"
nohup python index_queries.py -s workspace/datasets/train.csv > workspace/logs/index_queries.log &
