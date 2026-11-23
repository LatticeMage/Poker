FROM node:24

WORKDIR /app

# Run as the unprivileged `node` user (uid 1000) so files written into the
# bind-mounted source tree (.svelte-kit, build) are not owned by root.
RUN chown node:node /app
USER node

ARG BASE_PATH=""
ENV BASE_PATH=$BASE_PATH

COPY --chown=node:node package*.json .npmrc ./
RUN npm install

COPY --chown=node:node . .

# The static site ends up in /app/build (the CI workflow copies it out).
RUN npm run build

CMD ["npm", "run", "preview", "--", "--host", "0.0.0.0", "--port", "3000"]
