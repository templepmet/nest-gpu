echo "UID=$(id -u $USER)" > .devcontainer/.env
echo "GID=$(id -g $USER)" >> .devcontainer/.env
echo "HOST_USER=$USER" >> .devcontainer/.env
echo "GIT_USER=$GIT_USER" >> .devcontainer/.env
echo "GIT_TOKEN=$GIT_TOKEN" >> .devcontainer/.env
