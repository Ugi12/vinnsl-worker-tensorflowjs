##FROM node:10

# production env
#ARG NODE_ENV=production
#ENV NODE_ENV=${NODE_ENV}

# Create app directory
##WORKDIR /app

# Install app dependencies
# A wildcard is used to ensure both package.json AND package-lock.json are copied
# where available (npm@5+)
##COPY package.json /app

##RUN npm install
# If you are building your code for production
# RUN npm ci --only=production

# Bundle app source
##COPY . /app

##CMD node server.js


##EXPOSE 3000


FROM node:10

RUN mkdir -p /home/node/app/node_modules && chown -R node:node /home/node/app

WORKDIR /home/node/app

COPY package*.json ./

USER node

RUN npm install

COPY --chown=node:node . /home/node/app

EXPOSE 3000

CMD [ "node", "server.js" ]

