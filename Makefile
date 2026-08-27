# Two run paths:
#   docker-*  laptop deployable, exactly as shipped
#   dev-*     native processes on the H200 box, where Docker is not available
SHELL := /bin/bash
ROOT  := $(shell pwd)
PY    := /DATA2/home/parth/.conda/envs/cp-api/bin/python
MONGOD := $(ROOT)/vendor/mongodb-linux-x86_64-ubuntu2204-7.0.14/bin/mongod
REDISD := /DATA2/home/parth/.conda/envs/cp-api/bin/redis-server
RUN   := $(ROOT)/.run

.PHONY: docker-up docker-down docker-logs dev-up dev-down api test fmt

docker-up:
	docker compose up -d
docker-down:
	docker compose down
docker-logs:
	docker compose logs -f api

dev-up:
	mkdir -p $(RUN)/mongo $(RUN)/log
	$(MONGOD) --dbpath $(RUN)/mongo --bind_ip 127.0.0.1 --port 27817 \
		--wiredTigerCacheSizeGB 1 --fork --logpath $(RUN)/log/mongod.log
	$(REDISD) --port 6479 --bind 127.0.0.1 --daemonize yes \
		--maxmemory 256mb --maxmemory-policy allkeys-lru --save '' \
		--logfile $(RUN)/log/redis.log --pidfile $(RUN)/redis.pid

dev-down:
	-$(REDISD:-server=-cli) -p 6479 shutdown nosave 2>/dev/null
	-$(MONGOD) --dbpath $(RUN)/mongo --shutdown

api:
	set -a; . $(ROOT)/.env.native; set +a; \
	cd $(ROOT) && PYTHONPATH=$(ROOT) $(PY) -m uvicorn api.main:app \
		--host 0.0.0.0 --port $${API_PORT:-8080} --reload

test:
	cd $(ROOT) && PYTHONPATH=$(ROOT) $(PY) -m pytest api/ -q
